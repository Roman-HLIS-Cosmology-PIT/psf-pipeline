from scm_pipeline import PipelineStage
from scm_pipeline.data_types import (
    ASDFFile,
    ParquetFile,
    TextFile,
    YamlFile,
    Directory,
)
from .types import PiffFile


class DetectionStage(PipelineStage):
    """
    This pipeline element is for detecting objects via SEP

    Input:
    Exposure is the name of the file has the image for each SCA, weights, and badpix maps.

    Output:
    object_catalog is a catalog of all objects detected in the exposure on any SCA.
    """

    name = "DetectionStage"
    inputs = [("Exposure", ASDFFile),
             ]
    outputs = [("object_catalog", ParquetFile),
              ]
    config_options = {"detection_threshold": float}

    def run(self):
        # Retrieve configuration:
        my_config = self.config
        print("Here is my configuration :", my_config)

       
        filename = self.get_input("Exposure")
        print(f" DetectionStage reading from {filename}")

        # TODO
        blah = process(filename)

        filename = self.get_output("object_catalog")
        print(f" DetectionStage writing to {filename}")
        open(filename, "w").write(blah)

class SelectionStage(PipelineStage):
    """
    This pipeline element is for selecting stars for psf fitting

    Probably this will be done by matching to a Gaia catalog and selecting relatively bright
    stars with good matches in that catalog.

    Note: The minimal output is to just make a catalog with the stars, but it is probably
    useful to keep all objects in the output file and just flag which ones are stars.
    This would make it easier for Piff to potentially reject objects with close neighbors.
    Alternatively, we can put that logic in this stage and eliminate stars that are
    too close (e.g. 1 arcsec) to another object.

    Inputs:
    1. object_catalog is the catalog with all detected objects in the image
    2. truth_catalog is a catalog of known stars (probably a catalog of Gaia stars)

    Output:
    star_catalog is a catalog with good PSF stars flagged to be used by Piff
    """

    name = "SelectionStage"
    inputs = [("object_catalog", ParquetFile),
             ]
    outputs = [("star_catalog", ParquetFile),
              ]
    config_options = {"magnitude_cut": float}

    def run(self):
        # Retrieve configuration:
        my_config = self.config
        print("Here is my configuration :", my_config)

       
        filename = self.get_input("object_catalog")
        print(f" SelectionStage reading from {filename}")

        # TODO
        blah = process(filename)

        filename = self.get_output("star_catalog")
        print(f" SelectionStage writing to {filename}")
        open(filename, "w").write(blah)

class ColorAssignmentStage(PipelineStage):
    """
    This pipeline element is for psf fitting using Piff

    The goal of this stage is to figure out which SED from a provided SED library
    each star should use when fitting based on the observed colors of the stars.
    The colors (or other label) are added as an additional column in the star catalog.
    And a dict mapping from color/label to file name is output as a YAML file.

    For assigning colors, use photometry from SOC coadds, if no match
    is found assign mean/median color/SED.

    Inputs:
    1. star_catalog is the catalog with flux measurements
    2. sed_library is a directory of SED files that Piff can potentially use for the stars.

    Outputs:
    1. color_star_catalog is the same star catalog with labels added as an additional column
    2. sed_mapping is a YAML file storing the dict mapping color label -> sed filename
    """

    name = "ColorAssignmentStage"
    inputs = [("star_catalog", ParquetFile),
              ("sed_library", Directory)
             ]
    outputs = [("color_star_catalog", ParquetFile),
               ("sed_mapping", YAMLFile),
              ]
    config_options = {"magnitude_cut": float} # placeholder

    def run(self):
        # Retrieve configuration:
        my_config = self.config
        print("Here is my configuration :", my_config)

       
        filename = self.get_input("star_catalog")
        print(f" SelectionStage reading from {filename}")

        # TODO
        blah = process(filename)

        filename = self.get_output("color_star_catalog")
        print(f" SelectionStage writing to {filename}")
        open(filename, "w").write(blah)

class FittingStage(PipelineStage):
    """
    This pipeline element is for psf fitting using Piff

    This stage actually runs the Piff executable, piffify.

    Inputs:
    1. Exposure is the name of the file has the image for each SCA, weights, and badpix maps.
    2. color_star_catalog is the star catalog file with color labels
    3. sed_library is a directory of SED files that Piff can potentially use for the stars.
    4. sed_mapping is a YAML file storing a dict mapping color label -> sed filename
    5. piff_config is a YAML file with the configuration details telling Piff how to do the fit.

    Output:
    psf_model is the .piff file with the fitted PSF model.
    """
    import piff
    import yaml

    name = "FittingStage"
    inputs = [("Exposure", ASDFFile),
              ("color_star_catalog", ParquetFile),
              ("sed_library", Directory),
              ("sed_mapping", YAMLFile),
              ("piff_config", YAMLFile),
             ]
    outputs = [("psf_model", PiffFile)]

    def run(self):
        print(f" FittingStage reading from {self.config['Exposure']}")

        with open(self.get_input("piff_config")) as fin:
            config = yaml.safe_load(fin.read())

        config['input']['image_file_name'] = self.get_input("Exposure")
        config['input']['cat_file_name'] = self.get_input("color_star_catalog")
        config['input']['sed_file_name'] = self.get_input("sed_mapping")
        config['output']['file_name'] = self.get_output("psf_model")

        verbose = config.get('verbose', 1)
        logger = piff.config.setup_logger(verbose=verbose)

        print(f" Starting piffify")
        psf = piff.piffify(config, logger=logger)



if __name__ == "__main__":
    cls = PipelineStage.main()
