# Basic PSF Pipeline

Initial development of the PSF pipeline, which will be integrated into the broader scm pipeline. We have divided the pipeline in 4 stages:

- Detection: This stage runs the python version of Source Extractor (SEP) to detect objects in single exposures. Initial development of this stage is underway.
- Selection: This stage will apply selection cuts to identify the stars that will be used for PSF modeling and fitting. (Not developed yet)
- Color Assignment: This stage will assign colors (and SEDs using those colors) to those selected stars. In order to assignt accurate colors we will need accurate photometry, for which we have not converged on a final decision yet. However, using the photometry from the SOC was discussed as a potential option. (Not developed yet)
- Fitting: This stage will run PIFF for modeling and fitting of the PSF. (Not developed yet)

The detection_kernels folder currently has 4 kernels made for the Roman PSF, with FWHM slightly larger (1.2x and 1.5x options) that the Roman PSF.
