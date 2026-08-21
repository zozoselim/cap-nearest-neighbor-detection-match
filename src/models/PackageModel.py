from pydantic import Field
from typing import List, Optional, Union, Literal

from sdks.novavision.src.base.model import (
    Package,
    Detection,
    Inputs,
    Configs,
    Outputs,
    Response,
    Request,
    Output,
    Input,
    Config,
)


# -----------------------------------------------------------------------------
# Extended Detection Model
# -----------------------------------------------------------------------------

class QueryDetection(Detection):
    nearestTargetDistance: Optional[float] = None


# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------

class InputQueryDetections(Input):
    name: Literal["inputQueryDetections"] = "inputQueryDetections"
    value: List[Detection]
    type: Literal["list"] = "list"

    class Config:
        title = "Query Detections"


class InputTargetDetections(Input):
    name: Literal["inputTargetDetections"] = "inputTargetDetections"
    value: List[Detection]
    type: Literal["list"] = "list"

    class Config:
        title = "Target Detections"


# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

class OutputQueryDetections(Output):
    name: Literal["outputQueryDetections"] = "outputQueryDetections"
    value: List[QueryDetection]
    type: Literal["list"] = "list"

    class Config:
        title = "Query Detections"


class OutputMatchedQueryDetections(Output):
    name: Literal["outputMatchedQueryDetections"] = "outputMatchedQueryDetections"
    value: List[QueryDetection]
    type: Literal["list"] = "list"

    class Config:
        title = "Matched Query Detections"


class OutputMatchedTargetDetections(Output):
    name: Literal["outputMatchedTargetDetections"] = "outputMatchedTargetDetections"
    value: List[Detection]
    type: Literal["list"] = "list"

    class Config:
        title = "Matched Target Detections"


# -----------------------------------------------------------------------------
# Anchor Point Options
# -----------------------------------------------------------------------------

class OptionCenter(Config):
    name: Literal["CENTER"] = "CENTER"
    value: Literal["CENTER"] = "CENTER"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Center"


class OptionCenterLeft(Config):
    name: Literal["CENTER_LEFT"] = "CENTER_LEFT"
    value: Literal["CENTER_LEFT"] = "CENTER_LEFT"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Center Left"


class OptionCenterRight(Config):
    name: Literal["CENTER_RIGHT"] = "CENTER_RIGHT"
    value: Literal["CENTER_RIGHT"] = "CENTER_RIGHT"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Center Right"


class OptionTopCenter(Config):
    name: Literal["TOP_CENTER"] = "TOP_CENTER"
    value: Literal["TOP_CENTER"] = "TOP_CENTER"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Top Center"


class OptionTopLeft(Config):
    name: Literal["TOP_LEFT"] = "TOP_LEFT"
    value: Literal["TOP_LEFT"] = "TOP_LEFT"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Top Left"


class OptionTopRight(Config):
    name: Literal["TOP_RIGHT"] = "TOP_RIGHT"
    value: Literal["TOP_RIGHT"] = "TOP_RIGHT"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Top Right"


class OptionBottomLeft(Config):
    name: Literal["BOTTOM_LEFT"] = "BOTTOM_LEFT"
    value: Literal["BOTTOM_LEFT"] = "BOTTOM_LEFT"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Bottom Left"


class OptionBottomCenter(Config):
    name: Literal["BOTTOM_CENTER"] = "BOTTOM_CENTER"
    value: Literal["BOTTOM_CENTER"] = "BOTTOM_CENTER"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Bottom Center"


class OptionBottomRight(Config):
    name: Literal["BOTTOM_RIGHT"] = "BOTTOM_RIGHT"
    value: Literal["BOTTOM_RIGHT"] = "BOTTOM_RIGHT"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Bottom Right"


class OptionKeypoint(Config):
    name: Literal["KEYPOINT"] = "KEYPOINT"
    value: Literal["KEYPOINT"] = "KEYPOINT"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Keypoint"


AnchorPointOption = Union[
    OptionCenter,
    OptionCenterLeft,
    OptionCenterRight,
    OptionTopCenter,
    OptionTopLeft,
    OptionTopRight,
    OptionBottomLeft,
    OptionBottomCenter,
    OptionBottomRight,
    OptionKeypoint,
]


# -----------------------------------------------------------------------------
# Configs
# -----------------------------------------------------------------------------

class ConfigQueryPoint(Config):
    name: Literal["ConfigQueryPoint"] = "ConfigQueryPoint"
    value: AnchorPointOption = Field(default_factory=OptionCenter)
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Query Anchor Point"
        json_schema_extra = {
            "shortDescription": "Anchor point used for query detections. Default: CENTER."
        }


class ConfigTargetPoint(Config):
    name: Literal["ConfigTargetPoint"] = "ConfigTargetPoint"
    value: AnchorPointOption = Field(default_factory=OptionCenter)
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Target Anchor Point"
        json_schema_extra = {
            "shortDescription": "Anchor point used for target detections. Default: CENTER."
        }


class ConfigQueryKeypointName(Config):
    name: Literal["ConfigQueryKeypointName"] = "ConfigQueryKeypointName"
    value: Optional[str] = None
    type: Literal["string"] = "string"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Query Keypoint Name"
        json_schema_extra = {
            "shortDescription": "Used only when Query Anchor Point is KEYPOINT."
        }


class ConfigTargetKeypointName(Config):
    name: Literal["ConfigTargetKeypointName"] = "ConfigTargetKeypointName"
    value: Optional[str] = None
    type: Literal["string"] = "string"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Target Keypoint Name"
        json_schema_extra = {
            "shortDescription": "Used only when Target Anchor Point is KEYPOINT."
        }


class ConfigMaxDistance(Config):
    name: Literal["ConfigMaxDistance"] = "ConfigMaxDistance"
    value: Optional[int] = Field(default=None, ge=0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Maximum Match Distance"
        json_schema_extra = {
            "shortDescription": "Optional maximum match distance in pixels."
        }


# -----------------------------------------------------------------------------
# Input / Config / Output Groups
# -----------------------------------------------------------------------------

class NearestNeighborDetectionMatchInputs(Inputs):
    inputQueryDetections: InputQueryDetections
    inputTargetDetections: InputTargetDetections


class NearestNeighborDetectionMatchConfigs(Configs):
    configQueryPoint: ConfigQueryPoint = Field(default_factory=ConfigQueryPoint)
    configTargetPoint: ConfigTargetPoint = Field(default_factory=ConfigTargetPoint)
    configQueryKeypointName: ConfigQueryKeypointName = Field(default_factory=ConfigQueryKeypointName)
    configTargetKeypointName: ConfigTargetKeypointName = Field(default_factory=ConfigTargetKeypointName)
    configMaxDistance: ConfigMaxDistance = Field(default_factory=ConfigMaxDistance)


class NearestNeighborDetectionMatchOutputs(Outputs):
    outputQueryDetections: OutputQueryDetections
    outputMatchedQueryDetections: OutputMatchedQueryDetections
    outputMatchedTargetDetections: OutputMatchedTargetDetections


# -----------------------------------------------------------------------------
# Request / Response
# -----------------------------------------------------------------------------

class NearestNeighborDetectionMatchRequest(Request):
    inputs: Optional[NearestNeighborDetectionMatchInputs] = None
    configs: NearestNeighborDetectionMatchConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class NearestNeighborDetectionMatchResponse(Response):
    outputs: NearestNeighborDetectionMatchOutputs


# -----------------------------------------------------------------------------
# Executor
# -----------------------------------------------------------------------------

class NearestNeighborDetectionMatch(Config):
    name: Literal["NearestNeighborDetectionMatch"] = "NearestNeighborDetectionMatch"

    value: Union[
        NearestNeighborDetectionMatchRequest,
        NearestNeighborDetectionMatchResponse,
    ]

    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Nearest Neighbor Detection Match"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[NearestNeighborDetectionMatch]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"
        json_schema_extra = {
            "target": "value"
        }


# -----------------------------------------------------------------------------
# Package
# -----------------------------------------------------------------------------

class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["component"] = "component"
    name: Literal["NearestNeighborDetectionMatch"] = "NearestNeighborDetectionMatch"