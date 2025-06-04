// Written by Dor Verbin, October 2021
// This is based on: http://thenewcode.com/364/Interactive-Before-and-After-Video-Comparison-in-HTML5-Canvas
// With additional modifications based on: https://jsfiddle.net/7sk5k4gp/13/

function playVids3(videoId) {
    var videoMerge = document.getElementById(videoId + "Merge3");
    var vid = document.getElementById(videoId);
    console.log(videoId + "Merge3");

    var pos_array = new Array();
    pos_array[0] = 0.333;
    pos_array[1] = 0.666;
    var flag = false;
    var line_flag = new Array();
    line_flag[0] = false;
    line_flag[1] = false;
    var vidWidth = vid.videoWidth/3;
    var vidHeight = vid.videoHeight;

    var mergeContext = videoMerge.getContext("2d");

    
    if (vid.readyState > 3) {
        vid.play();

        function trackLocation(e) {
            // Normalize to [0, 1]
            bcr = videoMerge.getBoundingClientRect();
            position = ((e.pageX - bcr.x) / bcr.width);
        }
        function trackLocationTouch(e) {
            // Normalize to [0, 1]
            bcr = videoMerge.getBoundingClientRect();
            tmp = ((e.pageX - bcr.x) / bcr.width)
            if(flag == true){
                // console.log(line_flag[0], line_flag[1]);
                if(Math.abs(tmp - pos_array[0])<= 0.1 && Math.abs(tmp - pos_array[1])<= 0.1){
                    if(line_flag[0] == false && line_flag[1] == false){
                        if(Math.abs(tmp - pos_array[0]) < Math.abs(tmp - pos_array[1])){line_flag[0] = true; line_flag[1] = false;}
                        else{line_flag[1] = true; line_flag[0] = false;}
                    }
                    if(line_flag[0] == true){
                        pos_array[0] = tmp;
                        if(tmp>pos_array[1]) pos_array[1] = tmp;
                    }else{
                        pos_array[1] = tmp;
                        if(tmp<pos_array[0]) pos_array[0] = tmp;
                    }
                }else{
                    if(Math.abs(tmp - pos_array[0])<= 0.1) {
                        if(line_flag[0] == false && line_flag[1] == false){
                            pos_array[0] = tmp; line_flag[0] = true; line_flag[1] = false;
                        }else if(line_flag[0] == false){
                            if(tmp<pos_array[0]) pos_array[0] = tmp;
                        }else{
                            pos_array[0] = tmp;
                        }
                    }
                    if(Math.abs(tmp - pos_array[1])<= 0.1) {
                        if(line_flag[0] == false && line_flag[1] == false){
                            pos_array[1] = tmp; line_flag[1] = true; line_flag[0] = false;
                        }else if(line_flag[1] == false){
                            if(tmp>pos_array[1]) pos_array[1] = tmp;
                        }else{
                            pos_array[1] = tmp;
                        }
                    }
                }
            }
        }

        function setflag(e){
            // console.log(e);
            if(e.type == "mousedown") flag = true;
            else{
                flag = false;
                line_flag[0] = false;
                line_flag[1] = false;
            }
        }

        videoMerge.addEventListener("mousemove",  trackLocationTouch, false); 
        videoMerge.addEventListener("mousedown", setflag, false);
        videoMerge.addEventListener("mouseup", setflag, false);
        videoMerge.addEventListener("mouseleave", setflag, false);


        function drawLoop() {
            mergeContext.drawImage(vid, 0, 0, vidWidth, vidHeight, 0, 0, vidWidth, vidHeight);
            var colStart = (vidWidth * pos_array[0]).clamp(0.0, vidWidth);
            var colWidth = Math.abs((vidWidth * pos_array[1]) - (vidWidth * pos_array[0])).clamp(0.0, vidWidth);
            var colStart2 = (vidWidth * pos_array[1]).clamp(0.0, vidWidth);
            var colWidth2 = Math.abs(vidWidth - (vidWidth * pos_array[1])).clamp(0.0, vidWidth);
            mergeContext.drawImage(vid, colStart+vidWidth, 0, colWidth, vidHeight, colStart, 0, colWidth, vidHeight);
            mergeContext.drawImage(vid, colStart2+vidWidth*2, 0, colWidth2, vidHeight, colStart2, 0, colWidth2, vidHeight);
            requestAnimationFrame(drawLoop);

            
            var arrowLength = 0.06 * vidHeight;
            var arrowheadWidth = 0.025 * vidHeight;
            var arrowheadLength = 0.04 * vidHeight;
            var arrowPosY = vidHeight / 10;
            var arrowWidth = 0.007 * vidHeight;
            var currX_array = new Array();
            currX_array[0] = vidWidth * pos_array[0];
            currX_array[1] = vidWidth * pos_array[1];

            for (i=0; i<currX_array.length; ++i){
                // Draw circle
                mergeContext.arc(currX_array[i], arrowPosY, arrowLength*0.7, 0, Math.PI * 2, false);
                mergeContext.fillStyle = "#FFD79340";
                mergeContext.fill()
                //mergeContext.strokeStyle = "#444444";
                //mergeContext.stroke()
                
                // Draw border
                mergeContext.beginPath();
                mergeContext.moveTo(vidWidth*pos_array[i], 0);
                mergeContext.lineTo(vidWidth*pos_array[i], vidHeight);
                mergeContext.closePath()
                mergeContext.strokeStyle = "#AAAAAA";
                mergeContext.lineWidth = 5;            
                mergeContext.stroke();
                
                // Draw arrow
                mergeContext.beginPath();
                mergeContext.moveTo(currX_array[i], arrowPosY - arrowWidth/3);
                
                // Move right until meeting arrow head
                mergeContext.lineTo(currX_array[i] + arrowLength/3 - arrowheadLength/3, arrowPosY - arrowWidth/3);
                
                // Draw right arrow head
                mergeContext.lineTo(currX_array[i] + arrowLength/3 - arrowheadLength/3, arrowPosY - arrowheadWidth/3);
                mergeContext.lineTo(currX_array[i] + arrowLength/3, arrowPosY);
                mergeContext.lineTo(currX_array[i] + arrowLength/3 - arrowheadLength/3, arrowPosY + arrowheadWidth/3);
                mergeContext.lineTo(currX_array[i] + arrowLength/3 - arrowheadLength/3, arrowPosY + arrowWidth/3);
                
                // Go back to the left until meeting left arrow head
                mergeContext.lineTo(currX_array[i] - arrowLength/3 + arrowheadLength/3, arrowPosY + arrowWidth/3);
                
                // Draw left arrow head
                mergeContext.lineTo(currX_array[i] - arrowLength/3 + arrowheadLength/3, arrowPosY + arrowheadWidth/3);
                mergeContext.lineTo(currX_array[i] - arrowLength/3, arrowPosY);
                mergeContext.lineTo(currX_array[i] - arrowLength/3 + arrowheadLength/3, arrowPosY  - arrowheadWidth/3);
                mergeContext.lineTo(currX_array[i] - arrowLength/3 + arrowheadLength/3, arrowPosY);
                
                mergeContext.lineTo(currX_array[i] - arrowLength/3 + arrowheadLength/3, arrowPosY - arrowWidth/3);
                mergeContext.lineTo(currX_array[i], arrowPosY - arrowWidth/3);
                
                mergeContext.closePath();
                
                mergeContext.fillStyle = "#AAAAAA";
                mergeContext.fill();
            }
        }
        requestAnimationFrame(drawLoop);
    } 
}

Number.prototype.clamp = function(min, max) {
  return Math.min(Math.max(this, min), max);
};
    
    
function resizeAndPlay3(element)
{
  var cv = document.getElementById(element.id + "Merge3");
  cv.width = element.videoWidth/3;
  cv.height = element.videoHeight;
  element.play();
  element.style.height = "0px";  // Hide video without stopping it
    
  playVids3(element.id);
}
