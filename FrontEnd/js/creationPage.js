var chosenImage = null;
const generate3Dbutton = document.getElementById("generatePly");

document.getElementById("images").addEventListener("click", function (el) {
  if (el.target && el.target.matches(".box")) {
    console.log("Box clicked");
    toggleChosenImage(el.target);
  } else if (el.target && el.target.nodeName == "IMG") {
    parentBox = el.target.parentNode;
    console.log("Image clicked");
    toggleChosenImage(parentBox);
  }
});

function toggleChosenImage(element) {
  if (chosenImage == null) {
    //If Image is null, simply define the chosen image
    chosenImage = element;
    chosenImage.classList.toggle("chosenImage");
    generate3Dbutton.classList.remove("inactive");
  } else if (chosenImage == element) {
    //If the user clicks the same image, interpret it as 'un-choosing' the image
    chosenImage.classList.toggle("chosenImage");
    chosenImage = null;
    generate3Dbutton.classList.add("inactive");
  } else {
    //In this scenario the user chooses another image.
    chosenImage.classList.toggle("chosenImage");
    chosenImage = element;
    chosenImage.classList.toggle("chosenImage");
  }
}

async function create3D() {
  if (chosenImage == null) {
    alert("Please choose an image");
    return;
  }

  job_id = null

  const src = chosenImage.getElementsByTagName("img")[0].getAttribute("src");
  imagePathJson = { imagePath: src };
  console.log(imagePathJson);
  time = src.split("/Images/").pop()
  /*if(!confirm("Are you sure you want to generate from image: " + "\n" + time)){
    return;
  }*/

  let waitingWindow = document.createElement("div");
  let waitingSpinner = document.createElement("div");
  let waitingText = document.createElement("p");
  let downlaodButton = document.createElement("button");

  waitingWindow.classList.add("waitingWindow");
  waitingSpinner.classList.add("spinnerWait");
  downlaodButton.classList.add("inactive");
  waitingText.classList.add("status")

  waitingWindow.appendChild(waitingText);
  waitingWindow.appendChild(waitingSpinner);
  waitingWindow.appendChild(downlaodButton);

  waitingText.textContent = "Current Status: ";
  waitingText.style.color = "white";
  downlaodButton.textContent= "Waiting for Download";
  waitingSpinner.style.display = "block";

  document.body.appendChild(waitingWindow)

  const response = await fetch("CreationPage", { //Calls generation
    method: "POST",
    body: JSON.stringify(imagePathJson),
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.message}`);
      } else {
        return response.json();
      }
    })
    .then((json) => {
      job_id = {"jobId": json.jobId}
    });

    var output = null;
    var stop = false;
    //Start while-loop to checking status and updating the waiting window.
    while (!stop) {
      await new Promise(r => setTimeout(r, 10000));
      const response = await fetch("CreationPage", {
        method: "POST",
        body: JSON.stringify(job_id), //give the job_id to the getStatus method
      });
      
      const status_json = await response.json();
      if(status_json.status == "COMPLETED"){ //If completed end
        console.log("YAY");
        output = status_json.output;
        stop = true;
      } else{ //If still in process sleep and continue
        console.log("Update status window");
        waitingText.textContent = "Current Status: " + status_json.status;
      }
    }

    waitingText.textContent = "Download is available"
    downlaodButton.classList.remove("inactive");
    downlaodButton.onclick = () => openDownload(output.download_url)
    waitingSpinner.style.display = "none";


}

function openDownload(download_url){
  window.location.replace(download_url)
}



//Method for calling the optimization logic in app.py
async function Optimize() {
  let input = document.getElementById("input").value; //Get prompt for optimization
  if (input.trim() == "") {
    //Check the textarea is not empty.
    alert("Please write something in the input field before optimizing");
    return;
  }
  let button = document.getElementById("optimize");
  let spinner = document.getElementById("optimizeSpinner");
  loading(button, spinner);

  let text = { prompt: document.getElementById("input").value }; //Put prompt into json format
  
  const response = await fetch("/CreationPage", {
    method: "POST",
    body: JSON.stringify(text),
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      } else {
        return response.json();
      }
    })
    .then((data) => {
      //Remove citation marks
      var optimizedResponse = data.split('"').join("").replace(/\\n/g, "\n"); //Gets the optimized prompt from app.py
      document.getElementById("output").value = optimizedResponse; //Put optimized prompt into output container.
    })
    .catch((error) => {
      stopLoading(button, spinner);
      console.error("Error:", error);
    });
  stopLoading(button, spinner);
}


//Method for calling generation method in app.py.
async function Generate() {
  console.log("**********Generate called**********");
  let input = document.getElementById("output").value; //Get the prompt
  if (input.trim() == "") {
    //Make sure it's not 'empty'
    alert("Please write something in the input field before optimizing");
    return;
  }
  let button = document.getElementById("generate");
  let spinner = document.getElementById("generateSpinner");
  loading(button, spinner); //Removes button and starts the loading icon.

  let images = document.getElementsByClassName("images")[0];
  let text = { imagePrompt: input }; //Define the prompt as a json to send to app.py
  const response = await fetch("/CreationPage", {
    method: "POST",
    body: JSON.stringify(text),
  })
    .then((response) => {
      //Checks response to see if there was a connection issue
      if (!response.ok) {
        stopLoading(button, spinner);
        alert("HTTP error! status: " + response.status);
        return;
      } else {
        return response.json();
      }
    })
    .then((imageJson) => {
      //If no connection issue, we take the response json we get from app.py
      if (imageJson.error == 1) {
        //See if there was an error with the google api call
        stopLoading(button, spinner);
        alert(imageJson.message);
        return;
      } else {
        console.log("Response received");
        let imagePath = imageJson.message; //Get the path to the saved image, that gemini created
        var image = document.createElement("img");
        image.src = imagePath; //After image is created, the path of the image that needs to be shown is set to the html image element
        console.log(imagePath);

        const box = document.createElement("div"); //Box for containing the image element.
        box.classList.add("box"); //Add class for styling and further logic (look at code at the top)

        box.appendChild(image); //Add image element to box
        images.appendChild(box); //Add box - with image inside - to the list of images
      }
    });
  stopLoading(button, spinner);
}

function loading(element, spinner) {
  //Starts loading animation;
  element.style.display = "none";
  spinner.style.display = "block";
}

function stopLoading(element, spinner) {
  //Stops loading animation.
  spinner.style.display = "none";
  element.style.display = "block";
}
