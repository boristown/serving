# TensorFlow Serving

[![Ubuntu Build Status](https://storage.googleapis.com/tensorflow-serving-kokoro-build-badges/ubuntu.svg)](https://storage.googleapis.com/tensorflow-serving-kokoro-build-badges/ubuntu.html)
[![Ubuntu Build Status at TF HEAD](https://storage.googleapis.com/tensorflow-serving-kokoro-build-badges/ubuntu-tf-head.svg)](https://storage.googleapis.com/tensorflow-serving-kokoro-build-badges/ubuntu-tf-head.html)
![Docker CPU Nightly Build Status](https://storage.googleapis.com/tensorflow-serving-kokoro-build-badges/docker-cpu-nightly.svg)
![Docker GPU Nightly Build Status](https://storage.googleapis.com/tensorflow-serving-kokoro-build-badges/docker-gpu-nightly.svg)

----
TensorFlow Serving is a flexible, high-performance serving system for
machine learning models, designed for production environments. It deals with
the *inference* aspect of machine learning, taking models after *training* and
managing their lifetimes, providing clients with versioned access via
a high-performance, reference-counted lookup table.
TensorFlow Serving provides out-of-the-box integration with TensorFlow models,
but can be easily extended to serve other types of models and data.

To note a few features:

-   Can serve multiple models, or multiple versions of the same model
    simultaneously
-   Exposes both gRPC as well as HTTP inference endpoints
-   Allows deployment of new model versions without changing any client code
-   Supports canarying new versions and A/B testing experimental models
-   Adds minimal latency to inference time due to efficient, low-overhead
    implementation
-   Features a scheduler that groups individual inference requests into batches
    for joint execution on GPU, with configurable latency controls
-   Supports many *servables*: Tensorflow models, embeddings, vocabularies,
    feature transformations and even non-Tensorflow-based machine learning
    models

## Serve a Tensorflow model in 60 seconds
```bash
# Download the TensorFlow Serving Docker image and repo
docker pull tensorflow/serving

git clone https://github.com/boristown/serving
# Location of demo models
TESTDATA="$(pwd)/serving/tensorflow_serving/servables/tensorflow/testdata"

# Start TensorFlow Serving container and open the REST API port
sudo docker run -t --rm -p 8500:8500 \
    -v "$TESTDATA/saved_model_turtle3:/models/turtle3" \
    -e MODEL_NAME=turtle3 \
    tensorflow/serving &

# Start TensorFlow Serving container and open the REST API port
sudo docker run -t --rm -p 8501:8501 \
    -v "$TESTDATA/saved_model_turtle5:/models/turtle5" \
    -e MODEL_NAME=turtle5 \
    tensorflow/serving &
    
# Query the model using the predict API
curl -d '{"instances": [0.9886316,0.9376678,0.9073673,1,0.9744808,0.8876629,0.9255857,0.922936,0.8919353,0.9329572,0.9157184,0.8919353,0.9150993,0.8998816,0.8841003,0.908493,0.9043286,0.8302168,0.8789388,0.8744934,0.8035839,0.925446,0.8449121,0.8070306,0.9423656,0.9183222,0.8552668,0.9036983,0.887709,0.8668583,0.9048287,0.9003055,0.865847,0.9194415,0.8740678,0.837139,0.9180428,0.899599,0.8707513,0.9053923,0.8963125,0.8013325,0.8473968,0.8161358,0.7889726,0.8561385,0.8224531,0.779869,0.9467762,0.7966362,0.7186664,0.9114476,0.9060623,0.822523,0.8538125,0.8406636,0.7870039,0.8169106,0.8070306,0.7665517,0.841397,0.806132,0.7658135,0.8278527,0.7781956,0.7252822,0.8449121,0.7951581,0.6206492,0.6611821,0.6212351,0.5748088,0.5884674,0.5816929,0.5541534,0.5695013,0.5547805,0.539131,0.5766378,0.5644875,0.5400693,0.569225,0.554536,0.5099292,0.5369384,0.5271934,0.4864001,0.4979345,0.4872686,0.4827819,0.5088544,0.4966723,0.4749992,0.4941272,0.4890626,0.4765138,0.4899501,0.4851141,0.4732257,0.4959499,0.4818912,0.4593592,0.5054504,0.4928619,0.4872321,0.5067586,0.5024688,0.4741212,0.5179596,0.5004191,0.4946512,0.5212333,0.5118598,0.4865065,0.5111454,0.4993935,0.4910536,0.5134315,0.5098578,0.4788857,0.53597,0.4998539,0.4865065,0.5439654,0.5351428,0.5011383,0.5249611,0.5032086,0.4930429,0.5249945,0.5199346,0.4684088,0.4823834,0.4699647,0.4392086,0.4917601,0.4815451,0.4655082,0.479324,0.4679309,0.4649017,0.5029721,0.4728415,0.4451719,0.508675,0.4982409,0.474561,0.4922046,0.4820912,0.4686279,0.4989792,0.4900279,0.4577953,0.4898025,0.4774759,0.4445384,0.4914886,0.457273,0.427406,0.5088544,0.4861604,0.4644238,0.5177817,0.5013241,0.4564791,0.5764092,0.5111977,0.4932064,0.5832536,0.5679819,0.5601007,0.5953658,0.5787494,0.5063824,0.5323548,0.5092131,0.5076017,0.5280079,0.5217477,0.4870495,0.5346316,0.5118058,0.4977185,0.543913,0.5268393,0.5063459,0.5559824,0.5397882,0.5018576,0.5601531,0.5504732,0.5247119,0.549862,0.5330265,0.4386291,0.5420015,0.517026,0.5022179,0.5442099,0.5411013,0.5203792,0.5521482,0.5388913,0.4993395,0.6253138,0.5040501,0.4699123,0.6318438,0.6129141,0.5725369,0.6090752,0.5914347,0.5580511,0.5962152,0.5935622,0.5700363,0.5985919,0.5872036,0.5257407,0.5863844,0.5737085,0.5536501,0.6073558,0.5654226,0.5654226,0.610104,0.5946339,0.5376084,0.5773888,0.5751184,0.5268393,0.6712812,0.5491745,0.4819642,0.6948214,0.6670485,0.5306878,0.5803751,0.5358652,0.5272998,0.587699,0.5802386,0.5131998,0.5651447,0.5314547,0.5183168,0.5671341,0.5600325,0.4761677,0.495553,0.4928984,0.4212284,0.4569046,0.4256199,0.3989441,0.4477184,0.4362191,0.3826277,0.5090337,0.3999395,0.39571,0.5097498,0.4866145,0.4678579,0.5034754,0.4842187,0.4424538,0.5556505,0.4649747,0.4199249,0.6034533,0.5448068,0.5092131,0.6094134,0.5910093,0.5370448,0.6204984,0.5816818,0.5356715,0.610339,0.5736736,0.5143413,0.5584162,0.5247039,0.4950116,0.6045012,0.5110374,0.4783571,0.6196109,0.5946767,0.5029721,0.5282555,0.5111803,0.4169385,0.4485155,0.4191883,0.3644522,0.4805067,0.3700471,0.299471,0.5547631,0.4022495,0.279249,0.6131158,0.551067,0.5381339,0.6340046,0.5949228,0.5799973,0.6373228,0.6264489,0.4598021,0.5300242,0.4824135,0.4383814,0.4773727,0.4581605,0.4091828,0.4473644,0.429435,0.3626153,0.4906741,0.4358079,0.4052629,0.4551932,0.4511684,0.4087272,0.4673435,0.4213411,0.3838835,0.5342442,0.3849092,0.3427553,0.4392086,0.4129996,0.1958351,0.2114481,0.2056595,0.1851502,0.2114481,0.2114275,0.1759036,0.1942792,0.1942792,0.1450285,0.1947333,0.1594444,0.1397845,0.1760354,0.152762,0.1330909,0.2106352,0.1506774,0.04602221,0.0746159,0.06739682,0.0001,0.02623522,0.0001,0.0001]}' \
    -X POST http://localhost:8500/v1/models/turtle3:predict

# Returns => { "predictions": [0.3364523947238922,0.6635476350784302] }

    
# Query the model5 using the predict API
curl -d '{"instances": [0.9886316,0.9376678,0.9073673,1,0.9744808,0.8876629,0.9255857,0.922936,0.8919353,0.9329572,0.9157184,0.8919353,0.9150993,0.8998816,0.8841003,0.908493,0.9043286,0.8302168,0.8789388,0.8744934,0.8035839,0.925446,0.8449121,0.8070306,0.9423656,0.9183222,0.8552668,0.9036983,0.887709,0.8668583,0.9048287,0.9003055,0.865847,0.9194415,0.8740678,0.837139,0.9180428,0.899599,0.8707513,0.9053923,0.8963125,0.8013325,0.8473968,0.8161358,0.7889726,0.8561385,0.8224531,0.779869,0.9467762,0.7966362,0.7186664,0.9114476,0.9060623,0.822523,0.8538125,0.8406636,0.7870039,0.8169106,0.8070306,0.7665517,0.841397,0.806132,0.7658135,0.8278527,0.7781956,0.7252822,0.8449121,0.7951581,0.6206492,0.6611821,0.6212351,0.5748088,0.5884674,0.5816929,0.5541534,0.5695013,0.5547805,0.539131,0.5766378,0.5644875,0.5400693,0.569225,0.554536,0.5099292,0.5369384,0.5271934,0.4864001,0.4979345,0.4872686,0.4827819,0.5088544,0.4966723,0.4749992,0.4941272,0.4890626,0.4765138,0.4899501,0.4851141,0.4732257,0.4959499,0.4818912,0.4593592,0.5054504,0.4928619,0.4872321,0.5067586,0.5024688,0.4741212,0.5179596,0.5004191,0.4946512,0.5212333,0.5118598,0.4865065,0.5111454,0.4993935,0.4910536,0.5134315,0.5098578,0.4788857,0.53597,0.4998539,0.4865065,0.5439654,0.5351428,0.5011383,0.5249611,0.5032086,0.4930429,0.5249945,0.5199346,0.4684088,0.4823834,0.4699647,0.4392086,0.4917601,0.4815451,0.4655082,0.479324,0.4679309,0.4649017,0.5029721,0.4728415,0.4451719,0.508675,0.4982409,0.474561,0.4922046,0.4820912,0.4686279,0.4989792,0.4900279,0.4577953,0.4898025,0.4774759,0.4445384,0.4914886,0.457273,0.427406,0.5088544,0.4861604,0.4644238,0.5177817,0.5013241,0.4564791,0.5764092,0.5111977,0.4932064,0.5832536,0.5679819,0.5601007,0.5953658,0.5787494,0.5063824,0.5323548,0.5092131,0.5076017,0.5280079,0.5217477,0.4870495,0.5346316,0.5118058,0.4977185,0.543913,0.5268393,0.5063459,0.5559824,0.5397882,0.5018576,0.5601531,0.5504732,0.5247119,0.549862,0.5330265,0.4386291,0.5420015,0.517026,0.5022179,0.5442099,0.5411013,0.5203792,0.5521482,0.5388913,0.4993395,0.6253138,0.5040501,0.4699123,0.6318438,0.6129141,0.5725369,0.6090752,0.5914347,0.5580511,0.5962152,0.5935622,0.5700363,0.5985919,0.5872036,0.5257407,0.5863844,0.5737085,0.5536501,0.6073558,0.5654226,0.5654226,0.610104,0.5946339,0.5376084,0.5773888,0.5751184,0.5268393,0.6712812,0.5491745,0.4819642,0.6948214,0.6670485,0.5306878,0.5803751,0.5358652,0.5272998,0.587699,0.5802386,0.5131998,0.5651447,0.5314547,0.5183168,0.5671341,0.5600325,0.4761677,0.495553,0.4928984,0.4212284,0.4569046,0.4256199,0.3989441,0.4477184,0.4362191,0.3826277,0.5090337,0.3999395,0.39571,0.5097498,0.4866145,0.4678579,0.5034754,0.4842187,0.4424538,0.5556505,0.4649747,0.4199249,0.6034533,0.5448068,0.5092131,0.6094134,0.5910093,0.5370448,0.6204984,0.5816818,0.5356715,0.610339,0.5736736,0.5143413,0.5584162,0.5247039,0.4950116,0.6045012,0.5110374,0.4783571,0.6196109,0.5946767,0.5029721,0.5282555,0.5111803,0.4169385,0.4485155,0.4191883,0.3644522,0.4805067,0.3700471,0.299471,0.5547631,0.4022495,0.279249,0.6131158,0.551067,0.5381339,0.6340046,0.5949228,0.5799973,0.6373228,0.6264489,0.4598021,0.5300242,0.4824135,0.4383814,0.4773727,0.4581605,0.4091828,0.4473644,0.429435,0.3626153,0.4906741,0.4358079,0.4052629,0.4551932,0.4511684,0.4087272,0.4673435,0.4213411,0.3838835,0.5342442,0.3849092,0.3427553,0.4392086,0.4129996,0.1958351,0.2114481,0.2056595,0.1851502,0.2114481,0.2114275,0.1759036,0.1942792,0.1942792,0.1450285,0.1947333,0.1594444,0.1397845,0.1760354,0.152762,0.1330909,0.2106352,0.1506774,0.04602221,0.0746159,0.06739682,0.0001,0.02623522,0.0001,0.0001]}' \
    -X POST http://localhost:8501/v1/models/turtle5:predict

# Returns => { "predictions": [0.3364523947238922,0.6635476350784302] }
```

## End-to-End Training & Serving Tutorial

Refer to the official Tensorflow documentations site for [a complete tutorial to train and serve a Tensorflow Model](https://www.tensorflow.org/tfx/tutorials/serving/rest_simple).


## Documentation

### Set up

The easiest and most straight-forward way of using TensorFlow Serving is with
Docker images. We highly recommend this route unless you have specific needs
that are not addressed by running in a container.

*   [Install Tensorflow Serving using Docker](tensorflow_serving/g3doc/docker.md)
    *(Recommended)*
*   [Install Tensorflow Serving without Docker](tensorflow_serving/g3doc/setup.md)
    *(Not Recommended)*
*   [Build Tensorflow Serving from Source with Docker](tensorflow_serving/g3doc/building_with_docker.md)
*   [Deploy Tensorflow Serving on Kubernetes](tensorflow_serving/g3doc/serving_kubernetes.md)

### Use

#### Export your Tensorflow model

In order to serve a Tensorflow model, simply export a SavedModel from your
Tensorflow program.
[SavedModel](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md)
is a language-neutral, recoverable, hermetic serialization format that enables
higher-level systems and tools to produce, consume, and transform TensorFlow
models.

Please refer to [Tensorflow documentation](https://www.tensorflow.org/guide/saved_model#save_and_restore_models)
for detailed instructions on how to export SavedModels.

#### Configure and Use Tensorflow Serving

* [Follow a tutorial on Serving Tensorflow models](tensorflow_serving/g3doc/serving_basic.md)
* [Configure Tensorflow Serving to make it fit your serving use case](tensorflow_serving/g3doc/serving_config.md)
* Read the [Performance Guide](tensorflow_serving/g3doc/performance.md)
and learn how to [use TensorBoard to profile and optimize inference requests](tensorflow_serving/g3doc/tensorboard.md)
* Read the [REST API Guide](tensorflow_serving/g3doc/api_rest.md)
or [gRPC API definition](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/apis)
* [Use SavedModel Warmup if initial inference requests are slow due to lazy initialization of graph](tensorflow_serving/g3doc/saved_model_warmup.md)
* [If encountering issues regarding model signatures, please read the SignatureDef documentation](tensorflow_serving/g3doc/signature_defs.md)
* If using a model with custom ops, [learn how to serve models with custom ops](tensorflow_serving/g3doc/custom_op.md)

### Extend

Tensorflow Serving's architecture is highly modular. You can use some parts
individually (e.g. batch scheduling) and/or extend it to serve new use cases.

* [Ensure you are familiar with building Tensorflow Serving](tensorflow_serving/g3doc/building_with_docker.md)
* [Learn about Tensorflow Serving's architecture](tensorflow_serving/g3doc/architecture.md)
* [Explore the Tensorflow Serving C++ API reference](https://www.tensorflow.org/tfx/serving/api_docs/cc/)
* [Create a new type of Servable](tensorflow_serving/g3doc/custom_servable.md)
* [Create a custom Source of Servable versions](tensorflow_serving/g3doc/custom_source.md)

## Contribute


**If you'd like to contribute to TensorFlow Serving, be sure to review the
[contribution guidelines](CONTRIBUTING.md).**


## For more information

Please refer to the official [TensorFlow website](http://tensorflow.org) for
more information.
