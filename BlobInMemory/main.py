
import io
import pandas as pd

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateEntry

# WillTest project id is ''

def csv_to_df(blob_service_client, csv_filename, container='mlsamples'):
    '''
    Grab csv file with image labels from blob storage and convert 
    to Pandas DataFrame

    Parameters
    ----------
    blob_service_client (BlobServiceClient object): client to interact with blob storage at account level
    csv_filename (str) : filename of the csv which contains image labels
    container (str) : name of the blob storage container for relevant images

    Returns
    -------
    df (Pandas DataFrame) : DataFrame mapping image filenames to grain-size labels
    '''

    # Get blob client to interact with blob storage
    blob_client = blob_service_client.get_blob_client(container, csv_filename)

    # Read csv from blob
    csv_bytes = io.BytesIO()

    csv = blob_client.download_blob().readinto(csv_bytes)

    csv_str = str(csv_bytes.getvalue())

    # Format csv for dataframe (NOTE: this is a bit rough--can be improved)
    csv_temp = csv_str.split('\\r\\n')

    csv = [i.split(',') for i in csv_temp][2:-1]

    new_csv = []

    for k in csv:
        filename = k[0]
        fixed_filename = filename[:-3]+'jpg'
        new_csv.append([fixed_filename, k[1]])

    cols = ['filename', 'label']

    # Convert to dataframe
    df = pd.DataFrame(new_csv, columns=cols)

    # Close connection to blob storage
    csv_bytes.close()

    return df


def get_tags(trainer, project):
    '''
    Get all existing tags from Custom Vision project

    Parameters
    ----------
    trainer (CustomVisionTrainingClient object) : client for interacting with Custom Vision
    project (Project object) : relevant Custom Vision project

    Returns
    -------
    tags_dict (dict) : Dictionary mapping each tag name to corresponding tag object
    '''

    # Get tags from project
    tags = trainer.get_tags(project.id)

    # Add tags to dict with name as key and tag object as value
    tags_dict = {}

    for t in tags:
        tags_dict[t.name] = t

    return tags_dict


def tag_images(metadata_df, blob_service_client, container_name, trainer, project, base_image_url="ExampleDataset1/", multi_label='off'):
    '''
    Convert images to format which can be uploaded to Custom Vision portal, and add tags/labels to each image

    Parameters
    ----------
    metadata_df (Pandas DataFrame) : DataFrame mapping image filenames to grain-size labels
    blob_service_client (BlobServiceClient object): client to interact with blob storage at account level
    container_name (str) : name of the blob storage container for relevant images
    trainer (CustomVisionTrainingClient object) : client for interacting with Custom Vision
    project (Project object) : relevant Custom Vision project
    base_image_url (str) : path to desired images in blob storage
    multi_label (str) : on/off to allow for more than one label per image

    Returns
    -------
    image_list (list of ImageFileCreateEntry objects) : List of tagged image objects
    '''

    # Constant values -- change in production
    file_column = 'filename'
    target_variable = 'label'

    # Initialize list and tags which are already in project
    image_list = []
    tags = get_tags(trainer, project)

    # Loop over all images and add tags
    for index, row in metadata_df.iterrows():

        # Single tag case
        if '/' not in row[target_variable]:
            grain_sizes = [row[target_variable]]
        
        # Multi-label case
        elif '/' in row[target_variable] and multi_label == 'on':

            # Process values for target variable
            grain_sizes_temp = row[target_variable].split('/')
            grain_sizes = [s.strip() for s in grain_sizes_temp]
        
        else: 
            continue

        tag_ids = []
        
        # Get tag objects for this image, or create a new one
        for size in grain_sizes:

            if size not in tags:
                try:
                    new_tag = trainer.create_tag(project.id, size)
                    tags[size] = new_tag
                except:
                    pass
            
            tag_ids.append(tags[size].id)

        filename = base_image_url + row[file_column]
        print(filename)

        try:
            # Create a blob client using the local file name as the name for the blob
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=filename)

            # Next we retrieve and tag images for uploading, holding them in memory
            stream = io.BytesIO()

            # Read image into memory stream
            image = blob_client.download_blob().readinto(stream)

            # Tag and add to image list
            image_list.append(ImageFileCreateEntry(name=filename, contents=bytearray(stream.getvalue()), tag_ids=tag_ids))

            stream.close()

        except:
            print('Trouble with this file')

    return image_list


def upload_tagged_images(trainer, project, image_list):
    '''
    Upload tagged images to Custom Vision

    Parameters
    ----------
    trainer (CustomVisionTrainingClient object) : client for interacting with Custom Vision
    project (Project object) : relevant Custom Vision project
    image_list (list of ImageFileCreateEntry objects) : List of tagged image objects
    '''

    upload_result = trainer.create_images_from_files(project.id, images=image_list)

    # If upload fails, print results
    if not upload_result.is_batch_successful:
        print("Image batch upload failed.")
        for image in upload_result.images:
            print("Image status: ", image.status)


if __name__ == '__main__':

    # Set Up
    ENDPOINT = "<INSERT ENDPOINT>"

    # Hard-coded for ease, but change later
    training_key = "<INSERT TRAINING KEY"
    prediction_key = "<INSERT PREDICTION KEY"

    trainer = CustomVisionTrainingClient(training_key, endpoint=ENDPOINT)

    # Currently hard-coded to reference 'WillTest' project
    project = trainer.get_project('<INSERT PROJECT ID')

    # Retrieve data from blob storage
    connect_str = '<INSERT BLOB STORAGE CONNECTION STRING>'
    container_name = "mlsamples"

    # Create the BlobServiceClient object which will be used to create a container client
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    csv_image_dict = {'Example1.csv': 'ExampleDataset1/',
                      'Example2.csv': 'ExampleDataset2/'}

    csv_filename = 'Example1-Phis.csv'

    print('Getting Metadata...')
    df = csv_to_df(blob_service_client, csv_filename)
    # print(df)

    size = 10
    list_of_dfs = [df.loc[i:i+size-1,:] for i in range(0, len(df),size)]
    print('Beginning tagging and upload of', str(len(list_of_dfs)*10), 'images')

    for d in list_of_dfs:

        # Tag Images
        print('Tagging Images...')
        image_list = tag_images(d, blob_service_client, container_name, trainer, project)

        # Upload to Custom Vision
        print('Uploading to Custom Vision Portal...')
        upload_tagged_images(trainer, project, image_list)
    
    import time

    # Train model
    print ("Training...")
    iteration = trainer.train_project(project.id)
    while (iteration.status != "Completed"):
        iteration = trainer.get_iteration(project.id, iteration.id)
        print ("Training status: " + iteration.status)
        time.sleep(3)

    publish_iteration_name = "classifyModel"

    prediction_resource_id = '<INSERT PREDICTION RESOURCE ID'

    # The iteration is now trained. Publish it to the project endpoint
    trainer.publish_iteration(project.id, iteration.id, publish_iteration_name, prediction_resource_id)
    print ("Your model is ready for prediction.")

    from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient

    # Now there is a trained endpoint that can be used to make a prediction
    predictor = CustomVisionPredictionClient(prediction_key, endpoint=ENDPOINT)

    print('')
    print('Running prediction on sample image...')
    with open("temp.jpg", "rb") as image_contents:
        results = predictor.classify_image(
            project.id, publish_iteration_name, image_contents.read())

        # Display the results.
        for prediction in results.predictions:
            print("\t" + prediction.tag_name +
                ": {0:.2f}%".format(prediction.probability * 100))