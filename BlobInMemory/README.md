
# Blob Storage to Custom Vision

This repository contains a sample [Azure Function](https://docs.microsoft.com/en-us/azure/azure-functions/) designed to move image data between Blob Storage and the Azure Custom Vision service. Some functionality such as this is necessary to operationalize the Custom Vision service due to the time-consuming nature of manual image-tagging and movement. 

This code represents only one possible approach to automating the process, and there are doubtless many others. 

## Blobs In Memory

For this project, I needed to use the Blob Storage SDK for Python rather than the Blob Storage bindings for Azure Functions. In order to get the images ready for Custom Vision, I also needed to hold them as a bytearray type in memory, rather than utilizing a local filesystem (which Azure Functions does not expose). While I did find this to be possible, I had some difficulty getting there. 

My approach involved reading the blobs into a stream of type io.BytesIO (built-in type for Python 3). I then converted this to bytes by calling the getvalue() method on the stream. Finally, I cast the bytes to bytearray (essentially the same as bytes, aside from mutability). 

These steps allowed me to move images from Blob Storage into the Custom Vision service with no reliance on a filesystem, only memory. For others hoping to work with blobs in memory, this approach may be useful. 

## Notes

+ Depending on the number of images you are moving, this could be extremely long running. Consider breaking this process out across multiple function calls. Azure Functions on a consumption plan are unable to run longer than 10 minutes. 

+ One benefit to this approach is that all image data is handled in-memory by the Function. However, the corresponding danger is exceeding the memory limits of your Function. Depending on image size, you may need to change the number of images which are uploaded in a given batch, so as to stay within this limit.

+ The code makes a couple assumptions about the structure of the images and csv file in blob storage. Specifically, it is currently assumed that all uploaded images are withing the same folder in storage. Providing this info via variables in the code should allow for this, but any other case may require a small adjustment. 


## Environment

In order to use this code, you must set the following as variables either within the Azure Functions environment or in a settings file if running locally. 

    "ENDPOINT": Custom Vision service endpoint

    "TRAINING_KEY": Custom Vision training key

    "PROJECT": Project ID for the project to upload images into

    "BLOB_CONNECTION_STRING": Connection string for the Azure Blob Store where images will be uploaded

    "CONTAINER": Name of the container within Blob Storage holding the relevant images

## Other Material

Below I have included some materials that were particularly useful for me in developing this project:

+ [Azure Functions Python Documentation](https://docs.microsoft.com/en-us/azure/azure-functions/functions-reference-python#environment-variables)

+ [Custom Vision Quickstart](https://docs.microsoft.com/en-us/azure/cognitive-services/custom-vision-service/python-tutorial) for Python

