async function makeRequests() {
  try {
//////////////////////////////////////////////////////////////////////////////////////
    document.getElementById("progress").textContent = "Listening... in 5 Secs";
    const response = await fetch("/listen", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ record: true }),
    });

    if (response.ok) {
      const responseData = await response.json();
      document.getElementById("progress").textContent = "WAV Created!";
      const audioPath = responseData.path;
      console.log(audioPath);
//////////////////////////////////////////////////////////////////////////////////////
   const createPreprocessingResponse = await fetch("/preprocessing", {
          data: audioPath,
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ audioPath }),
        });  
   const responsePreprocessing = await createPreprocessingResponse.json();
   console.log(responsePreprocessing.status);
   document.getElementById("progress").textContent = responsePreprocessing.status;
   if (responsePreprocessing.status == "Audio Duration More Than 5 Secs"){
       return;
   }
      
//////////////////////////////////////////////////////////////////////////////////////
   const speakerPredictResponse = await fetch("/speaker-predict", {
      data: "speakerPredict",
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ status: "Speaker Predicting" }),
        });  
        
  const responseSpeakerPredict = await speakerPredictResponse.json();
  console.log(responseSpeakerPredict.speaker);
  document.getElementById("progress").textContent = `Speaker : ${responseSpeakerPredict.speaker}`;
  if (responseSpeakerPredict.speaker == "Speaker Unidentified"){
      return;
  }
//////////////////////////////////////////////////////////////////////////////////////
   const wordsPredictResponse = await fetch("/word-predict", {
      data: "wordPredict",
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ status: "Words Predicting" }),
        });  
        
  const responseWordsPredict = await wordsPredictResponse.json();
  console.log(responseWordsPredict.transcribe);
  document.getElementById("progress").textContent = `Speaker : ${responseSpeakerPredict.speaker}\nTranscribe : ${responseWordsPredict.transcribe}`;
//////////////////////////////////////////////////////////////////////////////////////
  if(responseWordsPredict.transcribe.includes("nyalakan lampu satu nol satu") == true){
      document.getElementById("satu-nol-satu").src = "/static/pic_bulbon.gif"
  }else if(responseWordsPredict.transcribe.includes("matikan lampu satu nol satu") == true){
      document.getElementById("satu-nol-satu").src = "/static/pic_bulboff.gif"
  }else if(responseWordsPredict.transcribe.includes("nyalakan lampu satu nol dua") == true){
      document.getElementById("satu-nol-dua").src = "/static/pic_bulbon.gif"
  }else if(responseWordsPredict.transcribe.includes("matikan lampu satu nol dua") == true){
      document.getElementById("satu-nol-dua").src = "/static/pic_bulboff.gif"
  }else if(responseWordsPredict.transcribe.includes("nyalakan lampu dua nol satu") == true){
      document.getElementById("dua-nol-satu").src = "/static/pic_bulbon.gif"
  }else if(responseWordsPredict.transcribe.includes("matikan lampu dua nol satu") == true){
      document.getElementById("dua-nol-satu").src = "/static/pic_bulboff.gif"
  }else{
      document.getElementById("progress").textContent = `Speaker : ${responseSpeakerPredict.speaker}\nTranscribe : ${responseWordsPredict.transcribe}\nPerintah Tidak Ditemukan`
  }
//////////////////////////////////////////////////////////////////////////////////////


    } else {
      console.error("Request failed with status:", response.status);
    }
  } catch (error) {
    console.error("Error:", error);
  }
}

const recordButton = document.getElementById("record");
recordButton.addEventListener("click", makeRequests);