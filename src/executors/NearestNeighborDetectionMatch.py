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


VALID_ANCHORS = {
    "CENTER",
    "CENTER_LEFT",
    "CENTER_RIGHT",
    "TOP_CENTER",
    "TOP_LEFT",
    "TOP_RIGHT",
    "BOTTOM_LEFT",
    "BOTTOM_CENTER",
    "BOTTOM_RIGHT",
    "KEYPOINT",
}


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

        query_point = self.safe_get_param(
            "ConfigQueryPoint",
            "configQueryPoint",
        )

        target_point = self.safe_get_param(
            "ConfigTargetPoint",
            "configTargetPoint",
        )

        query_keypoint_name = self.safe_get_param(
            "ConfigQueryKeypointName",
            "configQueryKeypointName",
        )

        target_keypoint_name = self.safe_get_param(
            "ConfigTargetKeypointName",
            "configTargetKeypointName",
        )

        max_distance = self.safe_get_param(
            "ConfigMaxDistance",
            "configMaxDistance",
        )

        self.query_point = self.normalize_anchor(
            query_point,
            default="CENTER",
        )

        self.target_point = self.normalize_anchor(
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

    def safe_get_param(
        self,
        *names,
        default=None,
    ):
        """
        Bir parametre birden fazla isimle bulunabiliyorsa sırayla dener.

        Optional alanlar request içinde hiç bulunmuyorsa component'in
        düşmesini engeller.
        """

        for name in names:

            try:
                value = self.request.get_param(
                    name
                )
            except Exception:
                continue

            if value is not None:
                return value

        return default

    @staticmethod
    def is_placeholder(value) -> bool:
        """
        NovaVision'ın {{changeable}} gibi henüz gerçek değer almamış
        placeholder stringlerini tespit eder.
        """

        if not isinstance(
            value,
            str,
        ):
            return False

        value = value.strip()

        return (
            value.startswith("{{")
            and value.endswith("}}")
        )

    @staticmethod
    def unwrap_config_value(
        value,
        default=None,
    ):
        """
        NovaVision config wrapper'larından gerçek primitive değeri çıkarır.

        Örnek:
        Config -> value -> value -> primitive
        """

        current = value

        for _ in range(6):

            if current is None:
                return default

            if isinstance(
                current,
                (int, float, bool),
            ):
                return current

            if isinstance(
                current,
                str,
            ):

                current = current.strip()

                if not current:
                    return default

                if (
                    NearestNeighborDetectionMatch.is_placeholder(
                        current
                    )
                ):
                    return default

                return current

            if isinstance(
                current,
                dict,
            ):

                if "value" not in current:
                    return default

                current = current.get(
                    "value"
                )

                continue

            if hasattr(
                current,
                "value",
            ):

                current = current.value

                continue

            return default

        return default

    @staticmethod
    def normalize_anchor(
        value,
        default="CENTER",
    ) -> str:
        """
        Anchor dropdown seçimini güvenli biçimde çözer.

        NovaVision request örneği:

        {
            "name": "CENTER",
            "value": "{{changeable}}"
        }

        Burada gerçek seçim name alanındaki CENTER'dır.
        """

        if value is None:
            return default

        # -------------------------------------------------------------
        # Primitive string
        # -------------------------------------------------------------

        if isinstance(
            value,
            str,
        ):

            value = value.strip()

            if not value:
                return default

            if (
                NearestNeighborDetectionMatch.is_placeholder(
                    value
                )
            ):
                return default

            anchor = value.upper()

            if anchor in VALID_ANCHORS:
                return anchor

            raise ValueError(
                f"Unsupported anchor point: {value}"
            )

        # -------------------------------------------------------------
        # Dict
        # -------------------------------------------------------------

        if isinstance(
            value,
            dict,
        ):

            option_name = value.get(
                "name"
            )

            if isinstance(
                option_name,
                str,
            ):

                option_name = (
                    option_name
                    .strip()
                    .upper()
                )

                if option_name in VALID_ANCHORS:
                    return option_name

            inner_value = value.get(
                "value"
            )

            if inner_value is not None:

                return (
                    NearestNeighborDetectionMatch.normalize_anchor(
                        inner_value,
                        default=default,
                    )
                )

            return default

        # -------------------------------------------------------------
        # Pydantic / NovaVision model
        # -------------------------------------------------------------

        option_name = getattr(
            value,
            "name",
            None,
        )

        if isinstance(
            option_name,
            str,
        ):

            option_name = (
                option_name
                .strip()
                .upper()
            )

            if option_name in VALID_ANCHORS:
                return option_name

        inner_value = getattr(
            value,
            "value",
            None,
        )

        if inner_value is not None:

            return (
                NearestNeighborDetectionMatch.normalize_anchor(
                    inner_value,
                    default=default,
                )
            )

        return default

    @staticmethod
    def normalize_optional_string(
        value,
    ) -> Optional[str]:

        value = (
            NearestNeighborDetectionMatch.unwrap_config_value(
                value
            )
        )

        if value is None:
            return None

        value = str(
            value
        ).strip()

        if not value:
            return None

        if (
            NearestNeighborDetectionMatch.is_placeholder(
                value
            )
        ):
            return None

        return value

    @staticmethod
    def parse_max_distance(
        value,
    ) -> Optional[int]:

        value = (
            NearestNeighborDetectionMatch.unwrap_config_value(
                value
            )
        )

        if value is None:
            return None

        try:
            numeric_value = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ) as error:

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

        return int(
            numeric_value
        )

    # ---------------------------------------------------------------------
    # Object helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def get_value(
        obj,
        key,
        default=None,
    ):

        if obj is None:
            return default

        if isinstance(
            obj,
            dict,
        ):

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
    def detection_to_dict(
        detection,
    ) -> dict:

        if isinstance(
            detection,
            dict,
        ):

            return deepcopy(
                detection
            )

        if hasattr(
            detection,
            "model_dump",
        ):

            return deepcopy(
                detection.model_dump()
            )

        if hasattr(
            detection,
            "dict",
        ):

            return deepcopy(
                detection.dict()
            )

        raise TypeError(
            "Unsupported Detection type."
        )

    @staticmethod
    def normalize_detections(
        detections,
    ) -> List:

        if detections is None:
            return []

        if isinstance(
            detections,
            list,
        ):
            return detections

        return [
            detections
        ]

    # ---------------------------------------------------------------------
    # Bounding Box
    # ---------------------------------------------------------------------

    @staticmethod
    def get_bbox(
        detection,
    ):

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

        left = (
            NearestNeighborDetectionMatch.get_value(
                bbox,
                "left",
            )
        )

        top = (
            NearestNeighborDetectionMatch.get_value(
                bbox,
                "top",
            )
        )

        width = (
            NearestNeighborDetectionMatch.get_value(
                bbox,
                "width",
            )
        )

        height = (
            NearestNeighborDetectionMatch.get_value(
                bbox,
                "height",
            )
        )

        if (
            left is None
            or top is None
            or width is None
            or height is None
        ):

            raise ValueError(
                "boundingBox must contain left, top, width and height."
            )

        left = float(
            left
        )

        top = float(
            top
        )

        width = float(
            width
        )

        height = float(
            height
        )

        right = (
            left
            + width
        )

        bottom = (
            top
            + height
        )

        center_x = (
            left
            + width / 2.0
        )

        center_y = (
            top
            + height / 2.0
        )

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
    def has_keypoint_data(
        detection,
    ) -> bool:

        keypoints = (
            NearestNeighborDetectionMatch.get_value(
                detection,
                "keyPoints",
            )
        )

        return (
            keypoints is not None
        )

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
                f"{source_name} keypoint name must be provided "
                "when KEYPOINT is selected."
            )

        if not detections:
            return

        has_keypoints = any(
            NearestNeighborDetectionMatch.has_keypoint_data(
                detection
            )
            for detection in detections
        )

        if not has_keypoints:

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
        # Numeric keypoint index
        # -------------------------------------------------------------

        if keypoint_name.isdigit():

            index = int(
                keypoint_name
            )

            if index >= len(
                keypoints
            ):
                return None

            keypoint = keypoints[
                index
            ]

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

            if (
                cx is None
                or cy is None
            ):
                return None

            return (
                float(cx),
                float(cy),
            )

        # -------------------------------------------------------------
        # Named keypoint
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

            if (
                cx is None
                or cy is None
            ):
                return None

            return (
                float(cx),
                float(cy),
            )

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
    def get_detection_id(
        detection,
    ):

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

        if (
            query_id is not None
            and target_id is not None
            and query_id == target_id
        ):
            return True

        if (
            same_collection
            and query_index == target_index
        ):
            return True

        return False

    # ---------------------------------------------------------------------
    # Matching
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
                distance
                - minimum_distance
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
        # Keypoint validation
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
        # Prepare query output
        # -------------------------------------------------------------

        output_query_detections = [
            self.detection_to_dict(
                detection
            )
            for detection in query_detections
        ]

        matched_query_detections = []
        matched_target_detections = []

        # Sadece gerçekten aynı Python objesi ise aynı collection.
        same_collection = (
            self.input_query_detections
            is self.input_target_detections
        )

        # -------------------------------------------------------------
        # Nearest-neighbor matching
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

            output_query_detections[
                query_index
            ][
                "nearestTargetDistance"
            ] = nearest_distance

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
        # Outputs
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