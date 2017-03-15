#include "bubble_helper_progs.c"
#include "heating_helper_progs.c"

/*
  USAGE: find_HII_bubbles [-p <num of processors>] <redshift> [<ionization efficiency factor zeta>] [<Tvir_min> <ionizing mfp in ionized IGM> <Alpha, power law for ionization efficiency>]

  Program FIND_HII_BUBBLES reads in the halo list file from
  ../Output_files/Halo_lists/updated_halos_z%06.2f_%i_%.0fMpc,
  where the relavant parameters are taken from INIT_PARAMS.H.
  For each box pixel, it cycles through various bubble radii
  (taken from ANAL_PARAMS.H), until it finds the largest
  radius such that the enclosed collapsed mass fraction
  (obtained by summing masses from the halo list file of
  halos whose centers are within the bubble, or by taking
  the mean collapsed mass from conditional press-schechter)
  is larger than 1/HII_EFF_FACTOR (from ANAL_PARAMS.H)
  or the optional command line efficiency parameter

  If optional efficiency parameter is not passed, then the
  value of HII_EFF_FACTOR in ANAL_PARAMS.H is used.

  NOTE: the optional argument of thread number including the -p flag, MUST
  be the first two arguments.  If these are omitted, num_threads defaults
  to NUMCORES in INIT_PARAMS.H

  See ANAL_PARAMS.H for updated parameters and algorithms!

  Author: Andrei Mesinger
  Date: 01/10/07

  Support for Alpha power law scaling of zeta with halo mass added by Bradley Greig, 2016
*/

float *Fcoll;

void init_21cmMC_arrays() {

    Overdense_spline_GL_low = calloc(Nlow,sizeof(float));
    Fcoll_spline_GL_low = calloc(Nlow,sizeof(float));
    second_derivs_low_GL = calloc(Nlow,sizeof(float));
    Overdense_spline_GL_high = calloc(Nhigh,sizeof(float));
    Fcoll_spline_GL_high = calloc(Nhigh,sizeof(float));
    second_derivs_high_GL = calloc(Nhigh,sizeof(float));

    Fcoll = (float *) malloc(sizeof(float)*HII_TOT_FFT_NUM_PIXELS);

    // FOLD: other callocs
    xi_low = calloc((NGLlow+1),sizeof(float));
    wi_low = calloc((NGLlow+1),sizeof(float));

    xi_high = calloc((NGLhigh+1),sizeof(float));
    wi_high = calloc((NGLhigh+1),sizeof(float));

}

void destroy_21cmMC_arrays() {

    free(Fcoll);

    free(Overdense_spline_GL_low);
    free(Fcoll_spline_GL_low);
    free(second_derivs_low_GL);
    free(Overdense_spline_GL_high);
    free(Fcoll_spline_GL_high);
    free(second_derivs_high_GL);

    free(xi_low);
    free(wi_low);

    free(xi_high);
    free(wi_high);

    free(Mass_Spline);
    free(Sigma_Spline);
    free(dSigmadm_Spline);
    free(second_derivs_sigma);
    free(second_derivs_dsigma);
}


FILE *LOG;
unsigned long long SAMPLING_INTERVAL = (((unsigned long long)(HII_TOT_NUM_PIXELS / 1.0e6)) + 1); //used to sample density field to compute mean collapsed fraction

int main(int argc, char ** argv){

    // FOLD: var definitions
    char filename[300];
    FILE *F, *pPipe;
    float REDSHIFT, mass, R, xf, yf, zf, growth_factor, pixel_mass, cell_length_factor;
    float ave_N_min_cell, ION_EFF_FACTOR, M_MIN, ALPHA;
    int x,y,z, N_min_cell, LAST_FILTER_STEP, num_th, arg_offset, i,j,k;
    unsigned long long ct, ion_ct, sample_ct;
    float f_coll_crit, pixel_volume, density_over_mean, res_xH, Splined_Fcoll;
    float *xH, TVIR_MIN, MFP, xHI_from_xrays, std_xrays;
    fftwf_complex *Fcoll_unfiltered, *Fcoll_filtered;
    fftwf_plan plan;
    double global_xH, ave_xHI_xrays, ave_den, ST_over_PS, mean_f_coll_st, mean_f_coll_ps, f_coll, ave_fcoll;
    const gsl_rng_type * T;
    gsl_rng * r;
    i = 0;
    float nua, dnua, temparg;

    ALPHA =  EFF_FACTOR_PL_INDEX; // = 0

    // check arguments
    if ((argc>2) && (argv[1][0]=='-') && ((argv[1][1]=='p') || (argv[1][1]=='P'))) {
        // user specified num proc
        num_th = atoi(argv[2]);
        fprintf(stderr, "find_HII_bubbles_mod_fcoll: threading with user-specified %i threads\n", num_th);
        arg_offset = 2;
    } else {
        num_th = NUMCORES;
        fprintf(stderr, "find_HII_bubbles_mod_fcoll: threading with default %i threads\n", num_th);
        arg_offset = 0;
    }
    if (argc == (arg_offset+2)) {
        ION_EFF_FACTOR = HII_EFF_FACTOR; // use default from ANAL_PARAMS.H
        TVIR_MIN = ION_Tvir_MIN;
        MFP = R_BUBBLE_MAX;
    } else if (argc == (arg_offset+3)) { // just use parameter efficiency
        ION_EFF_FACTOR = atof(argv[arg_offset+2]); // use command line parameter
        TVIR_MIN = ION_Tvir_MIN;
        MFP = R_BUBBLE_MAX;
    } else if (argc == (arg_offset+5)) { // use all reionization command line parameters
        ION_EFF_FACTOR = atof(argv[arg_offset+2]);
        TVIR_MIN = atof(argv[arg_offset+3]);
        MFP = atof(argv[arg_offset+4]);
    } else if (argc == (arg_offset+6)) { // use all reionization command line parameters
        ION_EFF_FACTOR = atof(argv[arg_offset+2]);
        TVIR_MIN = atof(argv[arg_offset+3]);
        MFP = atof(argv[arg_offset+4]);
        ALPHA = atof(argv[arg_offset+5]);
    } else {
        fprintf(stderr, "USAGE: find_HII_bubbles <redshift> [<ionization efficiency factor zeta>] [<Tvir_min> <ionizing mfp in ionized IGM> <Alpha, power law for ionization efficiency>]\nAborting...\n");
        return -1;
    }

    // FOLD: check for error initialising fftwf threads
    if (fftwf_init_threads() == 0) {
        fprintf(stderr, "find_HII_bubbles_mod_fcoll: ERROR: problem initializing fftwf threads\nAborting\n.");
        return -1;
    }
    omp_set_num_threads(num_th);
    fftwf_plan_with_nthreads(num_th);
    init_ps();
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc(T);
    REDSHIFT = atof(argv[arg_offset+1]);
    growth_factor = dicke(REDSHIFT);
    pixel_volume = pow(BOX_LEN/(float)HII_DIM, 3);
    pixel_mass = RtoM(L_FACTOR * BOX_LEN / (float)HII_DIM);
    f_coll_crit = 1 / ION_EFF_FACTOR;
    cell_length_factor = L_FACTOR;

    init_21cmMC_arrays();


    // Set the minimum source mass:
    // FOLD: check virial values set correctly
    if ( (TVIR_MIN > 0) && (ION_M_MIN > 0) && (argc < 5)) {
        fprintf(stderr, "You have to \"turn-off\" either the ION_M_MIN or the ION_Tvir_MIN option in ANAL_PARAMS.H\nAborting...\n");
        free_ps(); return -1;
    }
    if (TVIR_MIN > 0) { // use the virial temperature for Mmin
        if (TVIR_MIN < 9.99999e3) { // neutral IGM
            M_MIN = TtoM(REDSHIFT, TVIR_MIN, 1.22);
        } else { // ionized IGM
            M_MIN = TtoM(REDSHIFT, TVIR_MIN, 0.6);
        }
    } else if (TVIR_MIN < 0) { // use the mass
        M_MIN = ION_M_MIN;
    }
    // check for WDM
    if (P_CUTOFF && ( M_MIN < M_J_WDM())) {
        fprintf(stderr, "The default Jeans mass of %e Msun is smaller than the scale supressed by the effective pressure of WDM.\n", M_MIN);
        M_MIN = M_J_WDM();
        fprintf(stderr, "Setting a new effective Jeans mass from WDM pressure supression of %e Msun\n", M_MIN);
    }

    // open log file
    system("mkdir ../Log_files");
    sprintf(filename, "../Log_files/HII_bubble_log_file_%d", getpid());
    LOG = fopen(filename, "w");
    // FOLD: check for error opening log file
    if (!LOG){
        fprintf(stderr, "find_HII_bubbles.c: Error opening log file\nAborting...\n");
        fftwf_cleanup_threads();
        free_ps(); return -1;
    }

    // allocate memory for the neutral fraction box
    xH = (float *) fftwf_malloc(sizeof(float)*HII_TOT_NUM_PIXELS);
    if (!xH){
        fprintf(stderr, "find_HII_bubbles.c: Error allocating memory for xH box\nAborting...\n");
        fprintf(LOG, "find_HII_bubbles.c: Error allocating memory for xH box\nAborting...\n");
        fclose(LOG); fftwf_cleanup_threads();
        free_ps(); return -1;
    }
    for (ct = 0; ct < HII_TOT_NUM_PIXELS; ct++) {
        xH[ct] = 1;
    }

    // lets check if we are going to bother with computing the inhmogeneous field at all...
    if (ALPHA == 0.) {
        mean_f_coll_st = FgtrM_st(REDSHIFT, M_MIN);
    }
    mean_f_coll_ps = FgtrM(REDSHIFT, M_MIN);
    if ((mean_f_coll_st / f_coll_crit < HII_ROUND_ERR)) { // way too small to ionize anything...
        // NOTE Maybe delete this, maybe don't delete this, maybe go ____ ________
        fprintf(stderr, "The ST mean collapse fraction is %e, which is much smaller than the effective critical collapse fraction of %e\n I will just declare everything to be neutral\n", mean_f_coll_st, f_coll_crit);
        fprintf(LOG, "The ST mean collapse fraction is %e, which is much smaller than the effective critical collapse fraction of %e\n I will just declare everything to be neutral\n", mean_f_coll_st, f_coll_crit);

        if (!USE_TS_IN_21CM) { // find the neutral fraction:
            init_heat();
            global_xH = 1 - xion_RECFAST(REDSHIFT, 0);;
            destruct_heat();
            for (ct = 0; ct < HII_TOT_NUM_PIXELS; ct++) {
                xH[ct] = global_xH;
            }
        }

        // print out the xH box
        switch(FIND_BUBBLE_ALGORITHM) {
            case 2:
                if (!USE_HALO_FIELD) {
                    sprintf(filename, "../Boxes/xH_nohalos_z%06.2f_nf%f_eff%.1f_effPLindex%.1f_HIIfilter%i_Mmin%.1e_RHIImax%.0f_%i_%.0fMpc", REDSHIFT, global_xH, ION_EFF_FACTOR, ALPHA, HII_FILTER, M_MIN, MFP, HII_DIM, BOX_LEN);
                }
                break;
            default:
                if (!USE_HALO_FIELD) {
                    sprintf(filename, "../Boxes/sphere_xH_nohalos_z%06.2f_nf%f_eff%.1f_effPLindex%.1f_HIIfilter%i_Mmin%.1e_RHIImax%.0f_%i_%.0fMpc", REDSHIFT, global_xH, ION_EFF_FACTOR, ALPHA, HII_FILTER, M_MIN, MFP, HII_DIM, BOX_LEN);
                }
        }
        F = fopen(filename, "wb");
        fprintf(LOG, "Neutral fraction is %f\nNow writing xH box at %s\n", global_xH, filename);
        fprintf(stderr, "Neutral fraction is %f\nNow writing xH box at %s\n", global_xH, filename);
        if (mod_fwrite(xH, sizeof(float)*HII_TOT_NUM_PIXELS, 1, F) != 1){
            fprintf(stderr, "find_HII_bubbles.c: Write error occured while writing xH box.\n");
            fprintf(LOG, "find_HII_bubbles.c: Write error occured while writing xH box.\n");
        }
        fclose(F); fclose(LOG); fftwf_free(xH); fftwf_cleanup_threads();
        free_ps(); return (int) (global_xH * 100);
    }


    // 1st allocate fftw memories

    // <Fcoll>
    Fcoll_unfiltered = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    Fcoll_filtered = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    if (!Fcoll_unfiltered || !Fcoll_filtered) {
        fprintf(stderr, "find_HII_bubbles_mod_fcoll: Error allocating memory for model Fcoll box\nAborting...\n");
        fprintf(LOG, "find_HII_bubbles_mod_fcoll: Error allocating memory for model Fcoll box\nAborting...\n");
        fftwf_free(xH); fclose(LOG);
        fftwf_cleanup_threads();
        free_ps();
        return -1;
    }
    fprintf(stderr, "Reading in model Fcoll box\n");
    fprintf(LOG, "Reading in model Fcoll box\n");
    // </Fcoll>

    // <Fcoll>
    sprintf(filename, "../Boxes/Fcoll_output_file_z%06.2f_%i_%.0fMpc", REDSHIFT, HII_DIM, BOX_LEN);
    F = fopen(filename, "rb");
    if (!F) {
        fprintf(stderr, "find_HII_bubbles_mod_fcoll: Unable to open file: %s\n", filename);
        fprintf(LOG, "find_HII_bubbles_mod_fcoll: Unable to open file: %s\n", filename);
        fftwf_free(xH); fclose(LOG); fftwf_free(Fcoll_unfiltered); fftwf_free(Fcoll_filtered);
        fftwf_cleanup_threads();
        free_ps();
        return -1;
    }
    for (i = 0; i < HII_DIM; i++) {
        for (j = 0; j < HII_DIM; j++) {
            for (k = 0; k < HII_DIM; k++) {
                if (fread((float *)Fcoll_unfiltered + HII_R_FFT_INDEX(i,j,k), sizeof(float), 1, F) != 1){
                    fprintf(stderr, "find_HII_bubbles_mod_fcoll: Read error occured while reading model Fcoll box.\n");
                    fprintf(LOG, "find_HII_bubbles_mod_fcoll: Read error occured while reading model Fcoll box.\n");
                    fftwf_free(xH); fclose(LOG); fftwf_free(Fcoll_unfiltered); fftwf_free(Fcoll_filtered);
                    fftwf_cleanup_threads(); fclose(F); free_ps();
                    return -1;
                }
            }
        }
    }
    fclose(F);

    // do the fft to get the k-space Fcoll field
    fprintf(LOG, "begin initial ffts, clock=%06.2f\n", (double)clock()/CLOCKS_PER_SEC);
    fflush(LOG);

    plan = fftwf_plan_dft_r2c_3d(HII_DIM, HII_DIM, HII_DIM, (float *)Fcoll_unfiltered, (fftwf_complex *)Fcoll_unfiltered, FFTW_ESTIMATE);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan); fftwf_cleanup();
    // remember to add the factor of VOLUME/TOT_NUM_PIXELS when converting from
    //  real space to k-space
    // Note: we will leave off factor of VOLUME, in anticipation of the inverse FFT below
    for (ct = 0; ct < HII_KSPACE_NUM_PIXELS; ct++) {
        Fcoll_unfiltered[ct] /= (HII_TOT_NUM_PIXELS+0.0);
    }
    fprintf(LOG, "end initial ffts, clock=%06.2f\n", (double)clock()/CLOCKS_PER_SEC);
    fflush(LOG);
    // </Fcoll>

    // loop through the filter radii (in Mpc)
    initialiseSplinedSigmaM(M_MIN, 1e16);
    R = fmin(MFP, L_FACTOR*BOX_LEN);
    LAST_FILTER_STEP = 0;
    while (!LAST_FILTER_STEP) { //-------------------------------------------------------------------------------------

        if ( ((R/DELTA_R_HII_FACTOR) <= (cell_length_factor*BOX_LEN/(float)HII_DIM)) || ((R/DELTA_R_HII_FACTOR) <= R_BUBBLE_MIN) ){
            LAST_FILTER_STEP = 1;
        }

        // <Fcoll>
        fprintf(LOG, "before memcpy, clock=%06.2f\n", (double)clock()/CLOCKS_PER_SEC);
        fflush(LOG);
        memcpy(Fcoll_filtered, Fcoll_unfiltered, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
        fprintf(LOG, "begin filter, clock=%06.2f\n", (double)clock()/CLOCKS_PER_SEC);
        fflush(LOG);
        // Smooth/filter the model Fcoll field on scale R:
        HII_filter(Fcoll_filtered, HII_FILTER, R); // HII_FILTER == 1 => k-space top-hat filter
        fprintf(LOG, "end filter, clock=%06.2f\n", (double)clock() / CLOCKS_PER_SEC);
        fflush(LOG);

        // FFT back to real space
        fprintf(LOG, "begin fft with R=%f, clock=%06.2f\n", R, (double)clock()/CLOCKS_PER_SEC);
        fflush(LOG);
        plan = fftwf_plan_dft_c2r_3d(HII_DIM, HII_DIM, HII_DIM, (fftwf_complex *)Fcoll_filtered, (float *)Fcoll_filtered, FFTW_ESTIMATE);
        fftwf_execute(plan);
        fftwf_destroy_plan(plan); fftwf_cleanup();
        fprintf(LOG, "end fft with R=%f, clock=%06.2f\n", R, (double)clock()/CLOCKS_PER_SEC);
        fflush(LOG);
        // </Fcoll>

        /* Check if this is the last filtering scale.  If so, we don't need Fcoll_unfiltered anymore (because it's in k-space). We'll re-read it to get the real-space field (instead of inverse FT), which we will use to se [sic] the residual neutral fraction */
        ST_over_PS = 0;
        f_coll = 0;

        if (LAST_FILTER_STEP) {
            // <Fcoll>
            sprintf(filename, "../Boxes/Fcoll_output_file_z%06.2f_%i_%.0fMpc", REDSHIFT, HII_DIM, BOX_LEN);
            F = fopen(filename, "rb");
            if (!F) {
                fprintf(stderr, "find_HII_bubbles_mod_fcoll: ERROR: unable to open file %s\n", filename);
                fprintf(LOG, "find_HII_bubbles_mod_fcoll: ERROR: unable to open file %s\n", filename);
                fftwf_free(xH); fclose(LOG); fftwf_free(Fcoll_unfiltered); fftwf_free(Fcoll_filtered);  fftwf_cleanup_threads();
                free_ps(); return -1;
            }
            for (i = 0; i < HII_DIM; i++) {
                for (j = 0; j < HII_DIM; j++) {
                    for (k = 0; k <HII_DIM; k++){
                        if (fread((float *)Fcoll_unfiltered + HII_R_FFT_INDEX(i,j,k), sizeof(float), 1, F)!=1) {
                            fprintf(stderr, "find_HII_bubbles_mod_fcoll: Read error occured while reading model Fcoll box.\n");
                            fprintf(LOG, "find_HII_bubbles_mod_fcoll: Read error occured while reading model Fcoll box.\n");
                            fftwf_free(xH); fclose(LOG); fftwf_free(Fcoll_unfiltered); fftwf_free(Fcoll_filtered); fftwf_cleanup_threads(); fclose(F);
                            free_ps(); return -1;
                        }
                    }
                }
            }
            fclose(F);
            // </Fcoll>

            temparg = 2*(pow(sigma_z0(M_MIN), 2) - pow(sigma_z0(RtoM(cell_length_factor*BOX_LEN/(float)HII_DIM)), 2) ) ;
            if (temparg < 0) {  // our filtering scale has become too small
                break;
            }
            erfc_denom_cell = sqrt(temparg);


            // renormalize the collapse fraction so that the mean matches ST,
            // since we are using the evolved (non-linear) density field
            sample_ct = 0;

            if (ALPHA == 0.) {
                for (x = 0; x < HII_DIM; x++) {
                    for (y = 0; y < HII_DIM; y++) {
                        for (z = 0; z < HII_DIM; z++) {
                            f_coll += Fcoll_unfiltered[HII_R_FFT_INDEX(x,y,z)];
                        }
                    }
                }
            }
            f_coll /= (double) HII_TOT_NUM_PIXELS;
            ST_over_PS = mean_f_coll_st / f_coll;
        }
        // else if !LAST_FILTER_STEP && we're operating on the density field
        else {
            fprintf(LOG, "begin f_coll normalization if, clock=%06.2f\n", (double)clock()/CLOCKS_PER_SEC);
            fflush(LOG);
            temparg = 2 * (pow(sigma_z0(M_MIN), 2) - pow(sigma_z0(RtoM(R)), 2)); // Eq. (14) in 21cmFAST
            if (temparg < 0) {  // our filtering scale has become too small
                break;
            }
	        erfc_denom = sqrt(temparg);

            // NOTE Fcoll still empty here (for first filter step)

            // renormalize the collapse fraction so that the mean matches ST,
            // since we are using the evolved (non-linear) density field
            sample_ct=0;

            for (x = 0; x < HII_DIM; x++) {
                for (y = 0; y < HII_DIM; y++) {
                    for (z = 0; z < HII_DIM; z++) {
                        f_coll += Fcoll_filtered[HII_R_FFT_INDEX(x,y,z)];
                    }
                }
            }
            f_coll /= (double) HII_TOT_NUM_PIXELS;
            ST_over_PS = mean_f_coll_st / f_coll;
            fprintf(LOG, "end f_coll normalization if, clock=%06.2f\n", (double)clock()/CLOCKS_PER_SEC);
            fflush(LOG);
        }
        //     fprintf(stderr, "Last filter %i, R_filter=%f, fcoll=%f, ST_over_PS=%f, mean normalized fcoll=%f\n", LAST_FILTER_STEP, R, f_coll, ST_over_PS, f_coll*ST_over_PS);


        // NOTE Fcoll is very smoothed here for the first filter step
        // sprintf(filename, "../Boxes/Fcoll_output_file_FIRST_z%06.2f_%i_%.0fMpc", REDSHIFT, HII_DIM, BOX_LEN);
        // fprintf(stderr, "Writing Fcoll box\n");
        // fprintf(LOG, "Writing Fcoll box\n");
        // F = fopen(filename, "wb");
        // mod_fwrite(Fcoll, sizeof(float)*HII_TOT_FFT_NUM_PIXELS, 1, F);
        // fclose(F);


        // TODO: set ST_over_PS to 1?

        /* ----------------  MAIN LOOP THROUGH THE BOX ----------------
        i.e., do [something] at that particular filter step (normalise f_coll?), and find the xH */
        fprintf(LOG, "start of main loop scroll, clock=%06.2f\n", (double)clock()/CLOCKS_PER_SEC);
        fflush(LOG);
        // now lets scroll through the filtered box
        ave_xHI_xrays = ave_den = ave_fcoll = std_xrays = 0;
        ion_ct = 0;
        for (x = 0; x < HII_DIM; x++) {
            for (y = 0; y < HII_DIM; y++) {
                for (z = 0; z < HII_DIM; z++) {
                    if (LAST_FILTER_STEP) {
                        f_coll = ST_over_PS * Fcoll_unfiltered[HII_R_FFT_INDEX(x,y,z)];
                    }
                    else {
                        f_coll = ST_over_PS * Fcoll_filtered[HII_R_FFT_INDEX(x,y,z)];
                    }

                    // adjust the denominator of the collapse fraction for the residual electron fraction in the neutral medium
                    xHI_from_xrays = 1;

                    if (f_coll > f_coll_crit) { // If ionised

                        if (FIND_BUBBLE_ALGORITHM == 2) {// center method
                            xH[HII_R_INDEX(x, y, z)] = 0;

                        } else if (FIND_BUBBLE_ALGORITHM == 1) {// sphere method
                            update_in_sphere(xH, HII_DIM, R/BOX_LEN, x/(HII_DIM+0.0), y/(HII_DIM+0.0), z/(HII_DIM+0.0));

                        } else {
                            fprintf(stderr, "Incorrect choice of find bubble algorithm: %i\nAborting...", FIND_BUBBLE_ALGORITHM);
                            fprintf(LOG, "Incorrect choice of find bubble algorithm: %i\nAborting...", FIND_BUBBLE_ALGORITHM);
                            fflush(NULL);
                            z=HII_DIM;y=HII_DIM,x=HII_DIM;R=0;
                        }
                    }

                    /* check if this is the last filtering step.
                     if so, assign partial ionizations to those cells which aren't fully ionized

                     "we allow for partially-ionized cells by setting the cell’s ionized fraction to ζfcoll(x,z,Rcell) at the last ﬁlter step for those cells which are not fully ionized" */
                    else if (LAST_FILTER_STEP && (xH[HII_R_INDEX(x, y, z)] > TINY)) {
                        if (!USE_HALO_FIELD) {
                            f_coll = ST_over_PS * Fcoll_filtered[HII_R_FFT_INDEX(x,y,z)];
                            if (f_coll>1) f_coll=1;
                            ave_N_min_cell = f_coll * pixel_mass*density_over_mean / M_MIN; // ave # of M_MIN halos in cell
                            if (ave_N_min_cell < N_POISSON){
                                // the collapsed fraction is too small, lets add poisson scatter in the halo number
                                N_min_cell = (int) gsl_ran_poisson(r, ave_N_min_cell);
                                f_coll = N_min_cell * M_MIN / (pixel_mass*density_over_mean);
                            }
                        }

                        if (f_coll > 1) {
                            f_coll = 1;
                        }
                        res_xH = xHI_from_xrays - f_coll * ION_EFF_FACTOR;
                        // and make sure fraction doesn't blow up for underdense pixels
                        if (res_xH < 0) {
                            res_xH = 0;
                        } else if (res_xH > 1) {
                            res_xH = 1;
                        }
                        xH[HII_R_INDEX(x, y, z)] = res_xH;
                    } // end partial ionizations at last filtering step


                } // k
            } // j
        } // i

        R /= DELTA_R_HII_FACTOR; // changes the filtering scale for the next step

        // NOTE Fcoll still the same here as before main loop through the box
    } // endwhile;

    // find the neutral fraction
    global_xH = 0;
    for (ct = 0; ct < HII_TOT_NUM_PIXELS; ct++) {
        global_xH += xH[ct];
    }
    global_xH /= (float)HII_TOT_NUM_PIXELS;

    // print out the xH box
    switch(FIND_BUBBLE_ALGORITHM) {
        case 2:
            if (!USE_HALO_FIELD) {
                sprintf(filename, "../Boxes/xH_nohalos_FIRST_z%06.2f_nf%f_eff%.1f_effPLindex%.1f_HIIfilter%i_Mmin%.1e_RHIImax%.0f_%i_%.0fMpc", REDSHIFT, global_xH, ION_EFF_FACTOR, ALPHA, HII_FILTER, M_MIN, MFP, HII_DIM, BOX_LEN);
            }
            break;
        default:
            if (!USE_HALO_FIELD) {
                sprintf(filename, "../Boxes/sphere_xH_nohalos_FIRST_z%06.2f_nf%f_eff%.1f_effPLindex%.1f_HIIfilter%i_Mmin%.1e_RHIImax%.0f_%i_%.0fMpc", REDSHIFT, global_xH, ION_EFF_FACTOR, ALPHA, HII_FILTER, M_MIN, MFP, HII_DIM, BOX_LEN);
            }
    }
    F = fopen(filename, "wb");
    if (!F) {
        fprintf(stderr, "find_HII_bubbles_mod_fcoll: ERROR: unable to open file %s for writing!\n", filename);
        fprintf(LOG, "find_HII_bubbles_mod_fcoll: ERROR: unable to open file %s for writing!\n", filename);
        global_xH = -1;
    } else {
        fprintf(LOG, "Neutral fraction is %f\nNow writing xH box at %s\n", global_xH, filename);
        fprintf(stderr, "Neutral fraction is %f\nNow writing xH box at %s\n", global_xH, filename);
        fflush(LOG);
        if (mod_fwrite(xH, sizeof(float)*HII_TOT_NUM_PIXELS, 1, F)!=1){
            fprintf(stderr, "find_HII_bubbles.c: Write error occured while writing xH box.\n");
            fprintf(LOG, "find_HII_bubbles.c: Write error occured while writing xH box.\n");
            global_xH = -1;
        }
        fclose(F);
    }

    // deallocate
    fftwf_cleanup_threads();
    gsl_rng_free (r);
    fftwf_free(xH);
    fclose(LOG);
    fftwf_free(Fcoll_unfiltered); fftwf_free(Fcoll_filtered);

    destroy_21cmMC_arrays();

    free_ps();
    return (int) (global_xH * 100);
}
