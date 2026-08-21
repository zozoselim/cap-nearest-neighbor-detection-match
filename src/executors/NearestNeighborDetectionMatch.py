import os
import sys
import math

from copy import deepcopy
from typing import List, Optional, Tuple


sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "../../../../",
    )
)


from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor


if __package__:
    from ..models.PackageModel import PackageModel
    from ..utils.response import build_response
else:
    from components.NearestNeighborDetectionMatch.src.models.PackageModel import (
        PackageModel,
    )

    from components.NearestNeighborDetectionMatch.src.utils.response import (
        build_response,
    )


TIE_EPSILON_PX = 1.0


class NearestNeighborDetectionMatch(Component):

    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)

        self.request.model = PackageModel(
            **self.request.data
        )

        # -----------------------------------------------------------------
        # Inputs
        # -----------------------------------------------------------------

        self.input_query_detections = self.request.get_param(
            "inputQueryDetections"
        )

        self.input_target_detections = self.request.get_param(
            "inputTargetDetections"
        )

        # -----------------------------------------------------------------
        # Configs
        # -----------------------------------------------------------------

        query_point = self.request.get_param(
            "ConfigQueryPoint"
        )

        target_point = self.request.get_param(
            "ConfigTargetPoint"
        )

        query_keypoint_name = self.request.get_param(
            "ConfigQueryKeypointName"
        )

        target_keypoint_name = self.request.get_param(
            "ConfigTargetKeypointName"
        )

        max_distance = self.request.get_param(
            "ConfigMaxDistance"
        )

        # Roboflow defaults:
        # query_point  = CENTER
        # target_point = CENTER

        self.query_point = self.unwrap_config_value(
            query_point,
            default="CENTER",
        )

        self.target_point = self.unwrap_config_value(
            target_point,
            default="CENTER",
        )

        self.query_keypoint_name = (
            self.normalize_optional_string(
                query_keypoint_name
            )
        )

        self.target_keypoint_name = (
            self.normalize_optional_string(
                target_keypoint_name
            )
        )

        self.max_distance = self.parse_max_distance(
            max_distance
        )

        # -----------------------------------------------------------------
        # Outputs
        # -----------------------------------------------------------------

        self.output_query_detections = []
        self.output_matched_query_detections = []
        self.output_matched_target_detections = []

    # ---------------------------------------------------------------------
    # Bootstrap
    # ---------------------------------------------------------------------

    @staticmethod
    def bootstrap(config: dict = None) -> dict:
        return {}

    # ---------------------------------------------------------------------
    # Config helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def unwrap_config_value(value, default=None):
        """
        NovaVision dropdown/config wrapper'larından gerçek değeri çıkarır.
        """

        current = value

        for _ in range(4):

            if current is None:
                return default

            if isinstance(
                current,
                (str, int, float, bool),
            ):
                break

            if isinstance(current, dict):

                if "value" not in current:
                    break

                current = current["value"]
                continue

            if hasattr(current, "value"):
                current = current.value
                continue

            break

        if current is None:
            return default

        if isinstance(current, str):
            current = current.strip()

            if not current:
                return default

        return current

    @staticmethod
    def normalize_optional_string(value) -> Optional[str]:
        value = (
            NearestNeighborDetectionMatch.unwrap_config_value(
                value
            )
        )

        if value is None:
            return None

        value = str(value).strip()

        if not value:
            return None

        return value

    @staticmethod
    def parse_max_distance(value) -> Optional[int]:
        value = (
            NearestNeighborDetectionMatch.unwrap_config_value(
                value
            )
        )

        if value is None or value == "":
            return None

        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Maximum Match Distance must be an integer."
            ) from error

        if numeric_value < 0:
            raise ValueError(
                "Maximum Match Distance cannot be negative."
            )

        if not numeric_value.is_integer():
            raise ValueError(
                "Maximum Match Distance must be an integer."
            )

        return int(numeric_value)

    # ---------------------------------------------------------------------
    # General model helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def get_value(obj, key, default=None):

        if isinstance(obj, dict):
            return obj.get(
                key,
                default,
            )

        return getattr(
            obj,
            key,
            default,
        )

    @staticmethod
    def detection_to_dict(detection) -> dict:
        """
        Detection sonucunu response için bağımsız dict haline getirir.
        """

        if isinstance(detection, dict):
            return deepcopy(detection)

        if hasattr(detection, "model_dump"):
            return deepcopy(
                detection.model_dump()
            )

        if hasattr(detection, "dict"):
            return deepcopy(
                detection.dict()
            )

        raise TypeError(
            "Unsupported Detection type."
        )

    @staticmethod
    def normalize_detections(detections) -> List:

        if detections is None:
            return []

        if isinstance(detections, list):
            return detections

        return [
            detections
        ]

    # ---------------------------------------------------------------------
    # Bounding Box
    # ---------------------------------------------------------------------

    @staticmethod
    def get_bbox(detection):

        bbox = (
            NearestNeighborDetectionMatch.get_value(
                detection,
                "boundingBox",
            )
        )

        if bbox is None:
            raise ValueError(
                "Detection does not contain a boundingBox."
            )

        return bbox

    @staticmethod
    def get_bbox_anchor(
        detection,
        anchor: str,
    ) -> Tuple[float, float]:

        bbox = (
            NearestNeighborDetectionMatch.get_bbox(
                detection
            )
        )

        left = float(
            NearestNeighborDetectionMatch.get_value(
                bbox,
                "left",
            )
        )

        top = float(
            NearestNeighborDetectionMatch.get_value(
                bbox,
                "top",
            )
        )

        width = float(
            NearestNeighborDetectionMatch.get_value(
                bbox,
                "width",
            )
        )

        height = float(
            NearestNeighborDetectionMatch.get_value(
                bbox,
                "height",
            )
        )

        right = left + width
        bottom = top + height

        center_x = left + width / 2.0
        center_y = top + height / 2.0

        anchor_points = {
            "CENTER": (
                center_x,
                center_y,
            ),

            "CENTER_LEFT": (
                left,
                center_y,
            ),

            "CENTER_RIGHT": (
                right,
                center_y,
            ),

            "TOP_CENTER": (
                center_x,
                top,
            ),

            "TOP_LEFT": (
                left,
                top,
            ),

            "TOP_RIGHT": (
                right,
                top,
            ),

            "BOTTOM_LEFT": (
                left,
                bottom,
            ),

            "BOTTOM_CENTER": (
                center_x,
                bottom,
            ),

            "BOTTOM_RIGHT": (
                right,
                bottom,
            ),
        }

        if anchor not in anchor_points:
            raise ValueError(
                f"Unsupported anchor point: {anchor}"
            )

        return anchor_points[
            anchor
        ]

    # ---------------------------------------------------------------------
    # Keypoints
    # ---------------------------------------------------------------------

    @staticmethod
    def has_keypoint_data(detection) -> bool:

        keypoints = (
            NearestNeighborDetectionMatch.get_value(
                detection,
                "keyPoints",
            )
        )

        return keypoints is not None

    @staticmethod
    def validate_keypoint_configuration(
        detections: List,
        anchor: str,
        keypoint_name: Optional[str],
        source_name: str,
    ):

        if anchor != "KEYPOINT":
            return

        if not keypoint_name:
            raise ValueError(
                f"{source_name} keypoint name must be "
                "provided when KEYPOINT is selected."
            )

        if not detections:
            return

        has_keypoint_data = any(
            NearestNeighborDetectionMatch.has_keypoint_data(
                detection
            )
            for detection in detections
        )

        if not has_keypoint_data:
            raise ValueError(
                f"KEYPOINT selected for {source_name}, "
                "but detections do not contain keyPoints."
            )

    @staticmethod
    def get_keypoint_anchor(
        detection,
        keypoint_name: Optional[str],
    ) -> Optional[Tuple[float, float]]:

        keypoints = (
            NearestNeighborDetectionMatch.get_value(
                detection,
                "keyPoints",
            )
        )

        if not keypoints:
            return None

        if keypoint_name is None:
            return None

        keypoint_name = str(
            keypoint_name
        ).strip()

        if not keypoint_name:
            return None

        # -------------------------------------------------------------
        # NovaVision extension:
        # Numeric keypoint index support
        # -------------------------------------------------------------

        if keypoint_name.isdigit():

            index = int(
                keypoint_name
            )

            if index < 0:
                return None

            if index >= len(keypoints):
                return None

            keypoint = keypoints[index]

            cx = (
                NearestNeighborDetectionMatch.get_value(
                    keypoint,
                    "cx",
                )
            )

            cy = (
                NearestNeighborDetectionMatch.get_value(
                    keypoint,
                    "cy",
                )
            )

            if cx is None or cy is None:
                return None

            return (
                float(cx),
                float(cy),
            )

        # -------------------------------------------------------------
        # Named keypoint support
        # -------------------------------------------------------------

        for keypoint in keypoints:

            candidate_name = (
                NearestNeighborDetectionMatch.get_value(
                    keypoint,
                    "name",
                )
                or
                NearestNeighborDetectionMatch.get_value(
                    keypoint,
                    "label",
                )
                or
                NearestNeighborDetectionMatch.get_value(
                    keypoint,
                    "classLabel",
                )
            )

            if candidate_name != keypoint_name:
                continue

            cx = (
                NearestNeighborDetectionMatch.get_value(
                    keypoint,
                    "cx",
                )
            )

            cy = (
                NearestNeighborDetectionMatch.get_value(
                    keypoint,
                    "cy",
                )
            )

            if cx is None or cy is None:
                return None

            return (
                float(cx),
                float(cy),
            )

        # Roboflow'da tek detection içinde keypoint bulunmazsa
        # o detection eşleşmeye katılmaz.
        return None

    @staticmethod
    def get_anchor_point(
        detection,
        anchor: str,
        keypoint_name: Optional[str] = None,
    ) -> Optional[Tuple[float, float]]:

        if anchor == "KEYPOINT":

            return (
                NearestNeighborDetectionMatch.get_keypoint_anchor(
                    detection=detection,
                    keypoint_name=keypoint_name,
                )
            )

        return (
            NearestNeighborDetectionMatch.get_bbox_anchor(
                detection=detection,
                anchor=anchor,
            )
        )

    # ---------------------------------------------------------------------
    # Distance
    # ---------------------------------------------------------------------

    @staticmethod
    def calculate_distance(
        point_a: Tuple[float, float],
        point_b: Tuple[float, float],
    ) -> float:

        x1, y1 = point_a
        x2, y2 = point_b

        return math.hypot(
            x2 - x1,
            y2 - y1,
        )

    # ---------------------------------------------------------------------
    # Self-match
    # ---------------------------------------------------------------------

    @staticmethod
    def get_detection_id(detection):
        """
        Roboflow detection_id kullanır.

        detectionId NovaVision/camelCase uyumluluğu için
        fallback olarak desteklenmektedir.
        """

        detection_id = (
            NearestNeighborDetectionMatch.get_value(
                detection,
                "detection_id",
            )
        )

        if detection_id is not None:
            return detection_id

        return (
            NearestNeighborDetectionMatch.get_value(
                detection,
                "detectionId",
            )
        )

    @staticmethod
    def is_same_detection(
        query_detection,
        target_detection,
        query_index: int,
        target_index: int,
        same_collection: bool,
    ) -> bool:

        query_id = (
            NearestNeighborDetectionMatch.get_detection_id(
                query_detection
            )
        )

        target_id = (
            NearestNeighborDetectionMatch.get_detection_id(
                target_detection
            )
        )

        # Roboflow self-match davranışı
        if (
            query_id is not None
            and target_id is not None
            and query_id == target_id
        ):
            return True

        # NovaVision fallback:
        # gerçekten aynı Python koleksiyonu iki porta
        # verilmişse aynı index kendisi kabul edilir.
        if (
            same_collection
            and query_index == target_index
        ):
            return True

        return False

    # ---------------------------------------------------------------------
    # Nearest-neighbor matching
    # ---------------------------------------------------------------------

    def find_nearest_targets(
        self,
        query_detection,
        query_index: int,
        target_detections: List,
        same_collection: bool,
    ):

        query_anchor = self.get_anchor_point(
            detection=query_detection,
            anchor=self.query_point,
            keypoint_name=self.query_keypoint_name,
        )

        if query_anchor is None:
            return [], None

        candidates = []

        for target_index, target_detection in enumerate(
            target_detections
        ):

            if self.is_same_detection(
                query_detection=query_detection,
                target_detection=target_detection,
                query_index=query_index,
                target_index=target_index,
                same_collection=same_collection,
            ):
                continue

            target_anchor = self.get_anchor_point(
                detection=target_detection,
                anchor=self.target_point,
                keypoint_name=self.target_keypoint_name,
            )

            if target_anchor is None:
                continue

            distance = self.calculate_distance(
                point_a=query_anchor,
                point_b=target_anchor,
            )

            # max_distance ranking işleminden önce uygulanır.
            if (
                self.max_distance is not None
                and distance > self.max_distance
            ):
                continue

            candidates.append(
                (
                    target_detection,
                    distance,
                )
            )

        if not candidates:
            return [], None

        minimum_distance = min(
            distance
            for _, distance in candidates
        )

        matched_targets = [
            target_detection
            for target_detection, distance in candidates
            if abs(
                distance - minimum_distance
            ) <= TIE_EPSILON_PX
        ]

        return (
            matched_targets,
            minimum_distance,
        )

    # ---------------------------------------------------------------------
    # Run
    # ---------------------------------------------------------------------

    def run(self):

        query_detections = self.normalize_detections(
            self.input_query_detections
        )

        target_detections = self.normalize_detections(
            self.input_target_detections
        )

        # -------------------------------------------------------------
        # Validate keypoint configuration
        # -------------------------------------------------------------

        self.validate_keypoint_configuration(
            detections=query_detections,
            anchor=self.query_point,
            keypoint_name=self.query_keypoint_name,
            source_name="Query",
        )

        self.validate_keypoint_configuration(
            detections=target_detections,
            anchor=self.target_point,
            keypoint_name=self.target_keypoint_name,
            source_name="Target",
        )

        # -------------------------------------------------------------
        # Prepare outputs
        # -------------------------------------------------------------

        output_query_detections = [
            self.detection_to_dict(
                detection
            )
            for detection in query_detections
        ]

        matched_query_detections = []
        matched_target_detections = []

        # Equality (==) kullanmıyoruz.
        # Sadece gerçekten aynı object ise aynı collection kabul edilir.
        same_collection = (
            self.input_query_detections
            is self.input_target_detections
            or
            query_detections
            is target_detections
        )

        # -------------------------------------------------------------
        # Matching
        # -------------------------------------------------------------

        for query_index, query_detection in enumerate(
            query_detections
        ):

            (
                nearest_targets,
                nearest_distance,
            ) = self.find_nearest_targets(
                query_detection=query_detection,
                query_index=query_index,
                target_detections=target_detections,
                same_collection=same_collection,
            )

            # Roboflow:
            # nearest_target_distance
            #
            # NovaVision naming convention:
            # nearestTargetDistance

            output_query_detections[
                query_index
            ][
                "nearestTargetDistance"
            ] = nearest_distance

            # matchedQuery ve matchedTarget aynı index'te
            # birbirine karşılık gelir.

            for target_detection in nearest_targets:

                matched_query_detections.append(
                    deepcopy(
                        output_query_detections[
                            query_index
                        ]
                    )
                )

                matched_target_detections.append(
                    self.detection_to_dict(
                        target_detection
                    )
                )

        # -------------------------------------------------------------
        # Store outputs
        # -------------------------------------------------------------

        self.output_query_detections = (
            output_query_detections
        )

        self.output_matched_query_detections = (
            matched_query_detections
        )

        self.output_matched_target_detections = (
            matched_target_detections
        )

        return build_response(
            context=self
        )


if __name__ == "__main__":
    Executor(
        sys.argv[1]
    ).run()