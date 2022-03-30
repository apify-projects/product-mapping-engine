const fs = require('fs');
const util = require('util');
const Apify = require('apify');
const { imageHash }= require('image-hash');
const promisifiedImageHash = util.promisify(imageHash);
var path = require('path');
var img_path = process.argv[2]
var output_file = process.argv[3]
const writeStream = fs.createWriteStream(output_file);
const pathName = writeStream.path;
const files = fs.readdirSync(img_path);

// Create images hashes of images in given input folder and store image hashes to given output folder
Apify.main(async () => {
	let hashes = [];
	let imgs = []
	let data = []

	for (let i = 0; i < files.length; i++) {
		var imgPath = path.join(img_path, files[i]);
		imgs.push(files[i]);
		await promisifiedImageHash(imgPath, 8, true).then((hash) => {
			hashes.push(hash);
		});
	data.push(imgs[i] + ';' + hashes[i])
	}
	data = JSON.stringify(data);
	fs.writeFileSync(pathName, data);
});

