const fs = require('fs');
const { imageHash }= require('image-hash');
var path = require('path');
const img_path = 'data/image_data/';
const files = fs.readdirSync(img_path);
const util = require('util');
const promisifiedImageHash = util.promisify(imageHash);
const writeStream = fs.createWriteStream('data/hashes.json');
const pathName = writeStream.path;

// This is the main Node.js source code file of your actor.

// Import Apify SDK. For more information, see https://sdk.apify.com/
const Apify = require('apify');

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

