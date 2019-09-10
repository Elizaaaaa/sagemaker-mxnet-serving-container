# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
from __future__ import absolute_import

import os
import boto3

import pytest
from sagemaker import session
from sagemaker import utils
from sagemaker.mxnet import MXNetModel

from timeout import timeout_and_delete_endpoint_by_name

RESOURCE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'resources'))
DEFAULT_HANDLER_PATH = os.path.join(RESOURCE_PATH, 'resnet50v2')
MODEL_PATH = os.path.join(DEFAULT_HANDLER_PATH, 'model.tar.gz')
SCRIPT_PATH = os.path.join(DEFAULT_HANDLER_PATH, 'model', 'code', 'mxnet_model_service.py')


@pytest.fixture(autouse=True)
def skip_if_no_accelerator(accelerator_type):
    if accelerator_type is None:
        pytest.skip('Skipping because accelerator type was not provided')

@pytest.mark.skip_if_no_accelerator()
def test_elastic_inference():
    endpoint_name = utils.unique_name_from_base('mx-p3-8x-resnet')
    instance_type = 'ml.p3.8xlarge'
    framework_version = '1.4.1'
    
    maeve_client = boto3.client("maeve","us-west-2", endpoint_url="https://maeve.loadtest.us-west-2.ml-platform.aws.a2z.com")
    runtime_client = boto3.client(
        "sagemaker-runtime", 
        "us-west-2",
        endpoint_url="https://maeveruntime.loadtest.us-west-2.ml-platform.aws.a2z.com")
    
    sagemaker_session = session.Session(sagemaker_client=maeve_client, sagemaker_runtime_client=runtime_client)

    with timeout_and_delete_endpoint_by_name(endpoint_name=endpoint_name,
                                             sagemaker_session=sagemaker_session,
                                             minutes=20):
        prefix = 'mxnet-serving/default-handlers'
        model_data = sagemaker_session.upload_data(path=MODEL_PATH, key_prefix=prefix)
        model = MXNetModel(model_data=model_data,
                           entry_point=SCRIPT_PATH,
                           role='arn:aws:iam::841569659894:role/sagemaker-access-role',
                           image='763104351884.dkr.ecr.us-west-2.amazonaws.com/mxnet-inference:1.4.1-gpu-py36-cu100-ubuntu16.04',
                           framework_version=framework_version,
                           py_version='py3',
                           sagemaker_session=sagemaker_session)

        predictor = model.deploy(initial_instance_count=1,
                                 instance_type=instance_type,
                                 endpoint_name=endpoint_name)

        output = predictor.predict([[1, 2]])
        assert [[4.9999918937683105]] == output


test_elastic_inference()
