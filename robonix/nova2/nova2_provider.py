"""Publish the route that the generated gRPC server actually serves."""

from dataclasses import replace

from robonix_api import Primitive
from robonix_api.atlas_types import GrpcParams
from robonix_api.lifecycle import generated_grpc_metadata


class Nova2Primitive(Primitive):
    def declare_capability(self, contract_id, endpoint, transport, params=None, **kwargs):
        # fb76dc9 qualifies the shared Driver route, but registers @grpc
        # handlers with short Python service names. Atlas clients use these
        # fields verbatim to bind the protobuf descriptor and dial the server.
        if (isinstance(params, GrpcParams) and params.proto_file == "robonix_contracts.proto"
                and "." not in params.service_name and "/" not in params.method):
            service, method = generated_grpc_metadata(params.service_name, params.method)
            params = replace(params, service_name=service, method=method)
        return super().declare_capability(contract_id, endpoint, transport, params=params, **kwargs)
