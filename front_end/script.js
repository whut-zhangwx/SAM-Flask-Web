// alert("Hello, JavaScript!")

var ipaddress = "http://127.0.0.1:5000"

// process input image, load an image file from local to front end
function preProcess(event) {
  console.log("preProcess() is called.")
  const imageFile = event.target.files[0];
  
  if(imageFile) {
    previewImage(imageFile)
    setImage(imageFile)
  } else {
    alert('Please select an image to upload.');
  }
}

// preview the image in the front
function previewImage(imageFile) {
  const reader = new FileReader();

  reader.onload = function(e) {
    const previewImage = document.getElementById('previewImage');
    previewImage.src = e.target.result;   // 将图片的Base64数据设置为img标签的src
    previewImage.style.display = 'block'; // 显示图片
    document.getElementById('imageContainer').style.visibility = 'visible'; // display the container
  };

  reader.readAsDataURL(imageFile); // 读取图片并转换为Data URL
}

// passing the image from front end to back end and useing predictor.set_image() to embed the image
function setImage(imageFile) {
  console.log("setImage() is called.")
  // const fileInput = document.getElementById('imageInput');
  // const file = fileInput.files[0];

  const formData = new FormData();
  formData.append('image', imageFile);

  fetch(ipaddress+'/predictor_set_image', {method: 'POST', body: formData})
  .then(response => response.json()) // response.json() -> data
  .then(data => {if(data.success === false) alert("Set Image Error In Back End")})
  .catch(error => console.error('Error:', error));
}

// Segment Image with point prompt
function segmentImage(event) {
  // get the click position [x, y]
  [xPos, yPos] = getClickPos(event);
  // pass the [x, y] as a prompt to back end to decode the embedding and get a mask
  decode(xPos, yPos);
}

// get the position of click on the image
function getClickPos(e){
  var xPage = e.pageX;
  var yPage = e.pageY;
  identifyImage = document.getElementById("previewImage");
  img_x = locationLeft(identifyImage);
  img_y = locationTop(identifyImage);
  var xPos = xPage-img_x;
  var yPos = yPage-img_y;
  // alert('X : ' + xPos + '\n' + 'Y : ' + yPos);
  // segmentImage(xPos, yPos)
  return [xPos, yPos];
}

//找到元素的屏幕位置
function locationLeft(element){
  offsetTotal = element.offsetLeft;
  scrollTotal = 0; //element.scrollLeft but we dont want to deal with scrolling - already in page coords
  if (element.tagName != "BODY"){
    if (element.offsetParent != null)
      return offsetTotal+scrollTotal+locationLeft(element.offsetParent);
  }
  return offsetTotal+scrollTotal;
}

//find the screen location of an element
function locationTop(element){
  offsetTotal = element.offsetTop;
  scrollTotal = 0; //element.scrollTop but we dont want to deal with scrolling - already in page coords
  if (element.tagName != "BODY"){
    if (element.offsetParent != null)
      return offsetTotal+scrollTotal+locationTop(element.offsetParent);
  }
  return offsetTotal+scrollTotal;
}

// pass the [x, y] position as a prompt to get a mask
function decode(x, y) {
  console.log("decode() is called.")
  // Post the [x, y] to back end and fetch an mask-image from Flask backend
  fetch(ipaddress + "/decode_embedding", {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({points: [[x, y]]})
  })
  .then(response => response.json())  // response -> json
  .then(data => {
    // 构造 base64 图像 URL
    const imageBase64 = `data:image/jpeg;base64,${data.image}`;
    // document.getElementById('previewImage').src = imageBase64;
    const image = document.getElementById('previewImage');
    image.src = imageBase64;
    // image.style.display = block; // display the image
  })
  .catch(error => {
    console.error('Error fetching the image:', error);
  });
}