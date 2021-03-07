from tensorflow.python.compiler.tensorrt import trt_convert as trt

# Conversion Parameters 
conversion_params = trt.TrtConversionParams(precision_mode=trt.TrtPrecisionMode.FP16)

converter = trt.TrtGraphConverterV2(
    input_saved_model_dir='data/model/saved',
    conversion_params=conversion_params)

# Converter method used to partition and optimize TensorRT compatible segments
converter.convert()

# Optionally, build TensorRT engines before deployment to save time at runtime
# Note that this is GPU specific, and as a rule of thumb, we recommend building at runtime

# Save the model to the disk 
converter.save('data/model/tensorrt')