from sdks.novavision.src.helper.package import PackageHelper


if __package__:
    from ..models.PackageModel import (
        PackageModel,
        PackageConfigs,
        ConfigExecutor,
        NearestNeighborDetectionMatchOutputs,
        NearestNeighborDetectionMatchResponse,
        NearestNeighborDetectionMatch,
        OutputQueryDetections,
        OutputMatchedQueryDetections,
        OutputMatchedTargetDetections,
    )
else:
    from components.NearestNeighborDetectionMatch.src.models.PackageModel import (
        PackageModel,
        PackageConfigs,
        ConfigExecutor,
        NearestNeighborDetectionMatchOutputs,
        NearestNeighborDetectionMatchResponse,
        NearestNeighborDetectionMatch,
        OutputQueryDetections,
        OutputMatchedQueryDetections,
        OutputMatchedTargetDetections,
    )


def build_response(context):
    output_query_detections = OutputQueryDetections(
        value=context.output_query_detections
    )

    output_matched_query_detections = (
        OutputMatchedQueryDetections(
            value=context.output_matched_query_detections
        )
    )

    output_matched_target_detections = (
        OutputMatchedTargetDetections(
            value=context.output_matched_target_detections
        )
    )

    outputs = NearestNeighborDetectionMatchOutputs(
        outputQueryDetections=output_query_detections,
        outputMatchedQueryDetections=output_matched_query_detections,
        outputMatchedTargetDetections=output_matched_target_detections,
    )

    response = NearestNeighborDetectionMatchResponse(
        outputs=outputs
    )

    executor = NearestNeighborDetectionMatch(
        value=response
    )

    config_executor = ConfigExecutor(
        value=executor
    )

    package_configs = PackageConfigs(
        executor=config_executor
    )

    package = PackageHelper(
        packageModel=PackageModel,
        packageConfigs=package_configs,
    )

    return package.build_model(context)