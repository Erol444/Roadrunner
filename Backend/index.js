const express = require('express');
var bodyParser=require('body-parser');
const app = express();
var multer  = require('multer');
var fs=require('fs');
var path=require('path');
const FILE_PATH = '';
// configure multer
const upload = multer({dest: 'uploads/'});

app.post('/score', (req,res) => {
    console.log(req);
    console.log("get");
    res.send("Hello");
});

app.get('/', (req,res) => {
    console.log(req);
    console.log("get123");
    res.send("Hello123");
});

app.post('/image', upload.single('image'), async (req, res) => {
    res.send({
        status: true,
        message: 'File is uploaded.',
    });
});

app.get('/image', (req, res) => {
    var filePath = __dirname + '/uploads/'+req.query.id;
    console.log(filePath);
    res.download(filePath)
});

//Setting up server
 var server = app.listen(process.env.PORT || 8080, function () {
    var port = server.address().port;
    console.log("App now running on port", port);
 });