
def download_images(dataset):
    """
    Download images for a product from pair of products from the dataset
    @param images_path: what folder to save the images to
    @param pair_index: index of the pair
    @param product_index: either 1 or 2, specifying which product from the pair should images be downloaded for
    @param pair: pair of products from the dataset
    @return: the amount of images that have been downloaded
    """
    for index, row in full_pairs_dataset.iterrows():
        downloaded_images = 0
        for potential_url in pair["image{}".format(product_index)]:
            if is_url(potential_url):
                try:
                    response = requests.get(potential_url, timeout=5)
                except:
                    print(response)
                    continue

                print(response)
                if response.ok:
                    if imghdr.what("", h=response.content) is None:
                        continue

                    image_file_path = os.path.join(
                        images_path,
                        'pair_{}_product_{}_image_{}.jpg'.format(pair_index, product_index, downloaded_images + 1)
                    )

                    image = cv2.imdecode(np.asarray(bytearray(response.content), dtype="uint8"), cv2.IMREAD_COLOR)

                    # decreasing the size of the images when needed to increase speed and preserve memory
                    width_resize_ratio = configuration.IMAGE_RESIZE_WIDTH / image.shape[1]
                    height_resize_ratio = configuration.IMAGE_RESIZE_HEIGHT / image.shape[0]
                    resize_ratio = min(width_resize_ratio, height_resize_ratio)
                    if resize_ratio < 1:
                        image = cv2.resize(image, (0, 0), fx=resize_ratio, fy=resize_ratio)

                    cv2.imwrite(image_file_path, image)
                    downloaded_images += 1

    return images

def upload_images_to_kvs(images, kvs_client):
    pass
