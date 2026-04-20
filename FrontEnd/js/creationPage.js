var chosenImage = null;

document.getElementById("images").addEventListener("click", function (el) {
  if (el.target && el.target.matches(".box")) {
    console.log("Box clicked")
    toggleChosenImage(el.target)
  } else if (el.target && el.target.nodeName == "IMG") {
    parentBox = el.target.parentNode;
    console.log("Image clicked")
    toggleChosenImage(parentBox)
  }
});

function toggleChosenImage(element) {
  if (chosenImage == null) {
    //If Image is null, simply define the chosen image
    chosenImage = element
    chosenImage.classList.toggle("chosenImage");
  } else if (chosenImage == element) {
    //If the user clicks the same image, interpret it as 'un-choosing' the image
    chosenImage.classList.toggle("chosenImage");
    chosenImage = null;
  } else {
    //In this scenario the user chooses another image.
    chosenImage.classList.toggle("chosenImage");
    chosenImage = element;
    chosenImage.classList.toggle("chosenImage");
  }
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
        return response.text();
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
        alert(imageJson.response);
        return;
      } else {
        console.log("Response received");
        let imagePath = imageJson.response; //Get the path to the saved image, that gemini created
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
