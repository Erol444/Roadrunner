const express = require('express');
const app = express();

app.get('/gps', (req,res) => {
    console.log(req);
    console.log("get");
    res.send("Hello");
});

app.get('/', (req,res) => {
    console.log(req);
    console.log("get123");
    res.send("Hello123");
});

//Setting up server
 var server = app.listen(process.env.PORT || 8080, function () {
    var port = server.address().port;
    console.log("App now running on port", port);
 });