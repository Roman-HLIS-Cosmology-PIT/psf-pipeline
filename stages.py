from scm_pipeline import PipelineStage
from scm_pipeline.data_types import ASDFFile, ParquetFile
from .types import TextFile, YamlFile


class DetectionStage(PipelineStage):
    """
    This pipeline element is for detecting objects via SEP
    """

    name = "DetectionStage"
    inputs = [("Exposure", ASDFFile)]
    outputs = [("object_catalog", ParquetFile)]
    config_options = {"detection_threshold": float}

    def run(self):
        # Retrieve configuration:
        my_config = self.config
        print("Here is my configuration :", my_config)

       
        filename = self.get_input("Exposure")
        print(f" DetectionStage reading from {filename}")
        blah = process(filename)

        filename = self.get_output("object_catalog")
        print(f" DetectionStage writing to {filename}")
        open(filename, "w").write(blah)

class SelectionStage(PipelineStage):
    """
    This pipeline element is for selecting stars for psf fitting
    """

    name = "SelectionStage"
    inputs = [("object_catalog", ParquetFile)]
    outputs = [("star_catalog", ParquetFile)]
    config_options = {"magnitude_cut": float}

    def run(self):
        # Retrieve configuration:
        my_config = self.config
        print("Here is my configuration :", my_config)

       
        filename = self.get_input("object_catalog")
        print(f" SelectionStage reading from {filename}")
        blah = process(filename)

        filename = self.get_output("star_catalog")
        print(f" SelectionStage writing to {filename}")
        open(filename, "w").write(blah)

class ColorAssignmentStage(PipelineStage):
    """
    This pipeline element is for psf fitting using PIFF
    Output is the same star catalog with colors(?) and index/filename
    to corresponding SED for each object.
    For assigning colors, use photometry from SOC coadds, if no match
    is found assign mean/median color/SED.
    """

    name = "ColorAssignmentStage"
    inputs = [("star_catalog", ParquetFile), ("sed_library", Directory)]
    outputs = [("color_star_catalog", ParquetFile)]
    config_options = {"magnitude_cut": float}

    def run(self):
        # Retrieve configuration:
        my_config = self.config
        print("Here is my configuration :", my_config)

       
        filename = self.get_input("star_catalog")
        print(f" SelectionStage reading from {filename}")
        blah = process(filename)

        filename = self.get_output("color_star_catalog")
        print(f" SelectionStage writing to {filename}")
        open(filename, "w").write(blah)

class FittingStage(PipelineStage):
    """
    This pipeline element is for psf fitting using PIFF
    """

    name = "FittingStage"
    inputs = [("Exposure", ASDFFile), ("color_star_catalog", ParquetFile),  ("sed_library", Directory)]
    outputs = [("psf_model", PiffFile)]
    config_options = {"magnitude_cut": float}

    def run(self):
        # Retrieve configuration:
        my_config = self.config
        print("Here is my configuration :", my_config)

       
        filename = self.get_input("object_catalog")
        print(f" SelectionStage reading from {filename}")
        blah = process(filename)

        filename = self.get_output("star_catalog")
        print(f" SelectionStage writing to {filename}")
        open(filename, "w").write(blah)




if __name__ == "__main__":
    cls = PipelineStage.main()
