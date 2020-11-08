const express = require('express');
const app = express();
var cors = require('cors')
app.use(cors())
var multer  = require('multer');
var storage = multer.diskStorage({
destination: function (req, file, cb) {
    cb(null, 'uploads')
},
filename: function (req, file, cb) {
    cb(null, file.fieldname + '-' + Date.now() + '.jpg')
}
})
var upload = multer({ storage: storage })

// Persistent data please
var events = [];
var scores = [];

// Get data
app.get('/getscore', (req, res) => res.send(scores));
app.get('/getevents', (req, res) => res.send(events));

// Score upload
app.get('/score', (req,res) => {
    scores.push(req.query)
    res.send("OK");
});

// Image upload/download
app.post('/image', upload.single('image'), async (req, res) => {
    events.push({lat: req.body.lat, long: req.body.long, img: req.file.filename});
    res.send('Image was uploaded');
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