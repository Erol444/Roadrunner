const express = require('express');
const app = express();
var cors = require('cors')
var multer  = require('multer');

app.use(cors())
const multerStorage = multer.diskStorage({
    destination: (req, file, cb) => {
      cb(null, './temp');
    },
    filename: (req, file, cb) => {
      const ext = file.mimetype.split('/')[1];
      cb(null, `image-${Date.now()}.${ext}`);
    }
  });

const upload = multer({ storage: multerStorage })

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
app.post('/postimage', upload.single('image'), (req, res) => {
    events.push({lat: req.body.lat, long: req.body.long, img: req.file.filename});
    res.send('Image was uploaded');
});

app.get('/image', (req, res) => {
    var filePath = __dirname + '/temp/'+req.query.id;
    console.log(filePath);
    res.download(filePath)
});

//Setting up server
 var server = app.listen(process.env.PORT || 8080, function () {
    var port = server.address().port;
    console.log("App now running on port", port);
 });