const express = require('express');
const app = express();
var multer  = require('multer');
var storage = multer.diskStorage({
destination: function (req, file, cb) {
    cb(null, 'uploads')
},
filename: function (req, file, cb) {
    cb(null, file.fieldname + '-' + Date.now() + '.jpg')
}
})
// const upload = multer({dest: 'uploads/'});
var upload = multer({ storage: storage })

var imgs = [];
var scores = [];

// Score upload
app.get('/score', (req,res) => {
    scores.push(req.query)
    res.send("OK");
});
//Score downlaod
app.get('/getscore', (req, res) => {
    res.send(scores);
});

// Image upload/download
app.post('/image', upload.single('image'), async (req, res) => {
    imgs.push({lat: req.body.lat, long: req.body.long, img: req.file.filename});
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