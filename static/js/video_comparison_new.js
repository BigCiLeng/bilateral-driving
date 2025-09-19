// Enhanced Video Comparison Functions
// Supports 2-way and 3-way video comparison with improved stability

// Utility function for clamping values
Number.prototype.clamp = function(min, max) {
    return Math.min(Math.max(this, min), max);
};

// Enhanced 2-way video comparison
function playVids(videoId) {
    const videoMerge = document.getElementById(videoId + "Merge");
    const vid = document.getElementById(videoId);
    
    if (!videoMerge || !vid) {
        console.error(`Video comparison elements not found for ${videoId}`);
        return;
    }

    let position = 0.5;
    const vidWidth = vid.videoWidth / 2;
    const vidHeight = vid.videoHeight;
    
    console.log(`Starting 2-way comparison for ${videoId}, video size: ${vid.videoWidth}x${vid.videoHeight}`);

    const mergeContext = videoMerge.getContext("2d");
    
    if (vid.readyState >= 3) {
        vid.play();

        // Mouse tracking
        function trackLocation(e) {
            const bcr = videoMerge.getBoundingClientRect();
            position = ((e.pageX - bcr.x) / bcr.width).clamp(0, 1);
        }

        // Touch tracking
        function trackLocationTouch(e) {
            const bcr = videoMerge.getBoundingClientRect();
            position = ((e.touches[0].pageX - bcr.x) / bcr.width).clamp(0, 1);
        }

        // Event listeners
        videoMerge.addEventListener("mousemove", trackLocation, false);
        videoMerge.addEventListener("touchmove", trackLocationTouch, false);

        // Drawing loop
        function drawLoop() {
            if (vid.readyState < 3) return;
            
            // Clear canvas
            mergeContext.clearRect(0, 0, vidWidth, vidHeight);
            
            // Draw left side (first video)
            mergeContext.drawImage(vid, 0, 0, vidWidth, vidHeight, 0, 0, vidWidth, vidHeight);
            
            // Draw right side (second video) based on position
            const colStart = (vidWidth * position).clamp(0, vidWidth);
            const colWidth = (vidWidth - (vidWidth * position)).clamp(0, vidWidth);
            mergeContext.drawImage(vid, colStart + vidWidth, 0, colWidth, vidHeight, colStart, 0, colWidth, vidHeight);
            
            // Draw divider line
            const lineX = vidWidth * position;
            mergeContext.beginPath();
            mergeContext.moveTo(lineX, 0);
            mergeContext.lineTo(lineX, vidHeight);
            mergeContext.strokeStyle = "#FFFFFF";
            mergeContext.lineWidth = 2;
            mergeContext.stroke();
            
            // Draw handle
            const handleY = vidHeight / 10;
            const handleSize = Math.min(vidHeight * 0.08, 20);
            
            mergeContext.beginPath();
            mergeContext.arc(lineX, handleY, handleSize, 0, Math.PI * 2);
            mergeContext.fillStyle = "rgba(255, 255, 255, 0.8)";
            mergeContext.fill();
            mergeContext.strokeStyle = "#000000";
            mergeContext.lineWidth = 2;
            mergeContext.stroke();
            
            // Draw arrows
            const arrowLength = handleSize * 0.8;
            const arrowWidth = handleSize * 0.3;
            
            // Left arrow
            mergeContext.beginPath();
            mergeContext.moveTo(lineX - arrowLength, handleY);
            mergeContext.lineTo(lineX - arrowLength/2, handleY - arrowWidth/2);
            mergeContext.lineTo(lineX - arrowLength/2, handleY + arrowWidth/2);
            mergeContext.closePath();
            mergeContext.fillStyle = "#000000";
            mergeContext.fill();
            
            // Right arrow
            mergeContext.beginPath();
            mergeContext.moveTo(lineX + arrowLength, handleY);
            mergeContext.lineTo(lineX + arrowLength/2, handleY - arrowWidth/2);
            mergeContext.lineTo(lineX + arrowLength/2, handleY + arrowWidth/2);
            mergeContext.closePath();
            mergeContext.fillStyle = "#000000";
            mergeContext.fill();
            
            requestAnimationFrame(drawLoop);
        }
        
        requestAnimationFrame(drawLoop);
    } else {
        console.warn(`Video ${videoId} not ready for comparison (readyState: ${vid.readyState})`);
    }
}

// Enhanced 3-way video comparison
function playVids3(videoId) {
    const videoMerge = document.getElementById(videoId + "Merge3");
    const vid = document.getElementById(videoId);
    
    if (!videoMerge || !vid) {
        console.error(`Video comparison elements not found for ${videoId}`);
        return;
    }

    let positions = [0.33, 0.67];
    let isDragging = false;
    let dragIndex = -1;
    const vidWidth = vid.videoWidth / 3;
    const vidHeight = vid.videoHeight;
    
    console.log(`Starting 3-way comparison for ${videoId}, video size: ${vid.videoWidth}x${vid.videoHeight}`);

    const mergeContext = videoMerge.getContext("2d");
    
    if (vid.readyState >= 3) {
        vid.play();

        // Mouse tracking
        function trackLocation(e) {
            const bcr = videoMerge.getBoundingClientRect();
            const mouseX = (e.pageX - bcr.x) / bcr.width;
            
            if (isDragging && dragIndex >= 0) {
                positions[dragIndex] = mouseX.clamp(0, 1);
                
                // Ensure positions don't overlap
                if (dragIndex === 0 && positions[0] >= positions[1]) {
                    positions[0] = positions[1] - 0.01;
                } else if (dragIndex === 1 && positions[1] <= positions[0]) {
                    positions[1] = positions[0] + 0.01;
                }
            } else {
                // Check which handle is being hovered
                const threshold = 0.05;
                if (Math.abs(mouseX - positions[0]) < threshold) {
                    dragIndex = 0;
                } else if (Math.abs(mouseX - positions[1]) < threshold) {
                    dragIndex = 1;
                } else {
                    dragIndex = -1;
                }
            }
        }

        // Touch tracking
        function trackLocationTouch(e) {
            const bcr = videoMerge.getBoundingClientRect();
            const touchX = (e.touches[0].pageX - bcr.x) / bcr.width;
            
            if (isDragging && dragIndex >= 0) {
                positions[dragIndex] = touchX.clamp(0, 1);
                
                // Ensure positions don't overlap
                if (dragIndex === 0 && positions[0] >= positions[1]) {
                    positions[0] = positions[1] - 0.01;
                } else if (dragIndex === 1 && positions[1] <= positions[0]) {
                    positions[1] = positions[0] + 0.01;
                }
            } else {
                // Check which handle is being hovered
                const threshold = 0.05;
                if (Math.abs(touchX - positions[0]) < threshold) {
                    dragIndex = 0;
                } else if (Math.abs(touchX - positions[1]) < threshold) {
                    dragIndex = 1;
                } else {
                    dragIndex = -1;
                }
            }
        }

        // Mouse events
        function startDrag(e) {
            isDragging = true;
            trackLocation(e);
        }

        function endDrag(e) {
            isDragging = false;
            dragIndex = -1;
        }

        // Event listeners
        videoMerge.addEventListener("mousemove", trackLocation, false);
        videoMerge.addEventListener("mousedown", startDrag, false);
        videoMerge.addEventListener("mouseup", endDrag, false);
        videoMerge.addEventListener("mouseleave", endDrag, false);
        
        videoMerge.addEventListener("touchmove", trackLocationTouch, false);
        videoMerge.addEventListener("touchstart", (e) => {
            e.preventDefault();
            isDragging = true;
            trackLocationTouch(e);
        }, false);
        videoMerge.addEventListener("touchend", (e) => {
            e.preventDefault();
            isDragging = false;
            dragIndex = -1;
        }, false);

        // Drawing loop
        function drawLoop() {
            if (vid.readyState < 3) return;
            
            // Clear canvas
            mergeContext.clearRect(0, 0, vidWidth, vidHeight);
            
            // Draw first segment (left video)
            mergeContext.drawImage(vid, 0, 0, vidWidth, vidHeight, 0, 0, vidWidth, vidHeight);
            
            // Draw second segment (middle video)
            const colStart1 = (vidWidth * positions[0]).clamp(0, vidWidth);
            const colWidth1 = ((vidWidth * positions[1]) - (vidWidth * positions[0])).clamp(0, vidWidth);
            mergeContext.drawImage(vid, colStart1 + vidWidth, 0, colWidth1, vidHeight, colStart1, 0, colWidth1, vidHeight);
            
            // Draw third segment (right video)
            const colStart2 = (vidWidth * positions[1]).clamp(0, vidWidth);
            const colWidth2 = (vidWidth - (vidWidth * positions[1])).clamp(0, vidWidth);
            mergeContext.drawImage(vid, colStart2 + vidWidth * 2, 0, colWidth2, vidHeight, colStart2, 0, colWidth2, vidHeight);
            
            // Draw divider lines
            const lineX1 = vidWidth * positions[0];
            const lineX2 = vidWidth * positions[1];
            
            mergeContext.beginPath();
            mergeContext.moveTo(lineX1, 0);
            mergeContext.lineTo(lineX1, vidHeight);
            mergeContext.strokeStyle = "#FFFFFF";
            mergeContext.lineWidth = 2;
            mergeContext.stroke();
            
            mergeContext.beginPath();
            mergeContext.moveTo(lineX2, 0);
            mergeContext.lineTo(lineX2, vidHeight);
            mergeContext.strokeStyle = "#FFFFFF";
            mergeContext.lineWidth = 2;
            mergeContext.stroke();
            
            // Draw handles
            const handleY = vidHeight / 10;
            const handleSize = Math.min(vidHeight * 0.08, 20);
            
            // First handle
            mergeContext.beginPath();
            mergeContext.arc(lineX1, handleY, handleSize, 0, Math.PI * 2);
            mergeContext.fillStyle = dragIndex === 0 ? "rgba(255, 255, 255, 1)" : "rgba(255, 255, 255, 0.8)";
            mergeContext.fill();
            mergeContext.strokeStyle = "#000000";
            mergeContext.lineWidth = 2;
            mergeContext.stroke();
            
            // Second handle
            mergeContext.beginPath();
            mergeContext.arc(lineX2, handleY, handleSize, 0, Math.PI * 2);
            mergeContext.fillStyle = dragIndex === 1 ? "rgba(255, 255, 255, 1)" : "rgba(255, 255, 255, 0.8)";
            mergeContext.fill();
            mergeContext.strokeStyle = "#000000";
            mergeContext.lineWidth = 2;
            mergeContext.stroke();
            
            // Draw arrows on handles
            const arrowLength = handleSize * 0.6;
            const arrowWidth = handleSize * 0.3;
            
            // Arrows for first handle
            mergeContext.beginPath();
            mergeContext.moveTo(lineX1 - arrowLength, handleY);
            mergeContext.lineTo(lineX1 - arrowLength/2, handleY - arrowWidth/2);
            mergeContext.lineTo(lineX1 - arrowLength/2, handleY + arrowWidth/2);
            mergeContext.closePath();
            mergeContext.fillStyle = "#000000";
            mergeContext.fill();
            
            mergeContext.beginPath();
            mergeContext.moveTo(lineX1 + arrowLength, handleY);
            mergeContext.lineTo(lineX1 + arrowLength/2, handleY - arrowWidth/2);
            mergeContext.lineTo(lineX1 + arrowLength/2, handleY + arrowWidth/2);
            mergeContext.closePath();
            mergeContext.fillStyle = "#000000";
            mergeContext.fill();
            
            // Arrows for second handle
            mergeContext.beginPath();
            mergeContext.moveTo(lineX2 - arrowLength, handleY);
            mergeContext.lineTo(lineX2 - arrowLength/2, handleY - arrowWidth/2);
            mergeContext.lineTo(lineX2 - arrowLength/2, handleY + arrowWidth/2);
            mergeContext.closePath();
            mergeContext.fillStyle = "#000000";
            mergeContext.fill();
            
            mergeContext.beginPath();
            mergeContext.moveTo(lineX2 + arrowLength, handleY);
            mergeContext.lineTo(lineX2 + arrowLength/2, handleY - arrowWidth/2);
            mergeContext.lineTo(lineX2 + arrowLength/2, handleY + arrowWidth/2);
            mergeContext.closePath();
            mergeContext.fillStyle = "#000000";
            mergeContext.fill();
            
            requestAnimationFrame(drawLoop);
        }
        
        requestAnimationFrame(drawLoop);
    } else {
        console.warn(`Video ${videoId} not ready for comparison (readyState: ${vid.readyState})`);
    }
}

// Enhanced resize and play functions
function resizeAndPlay(element) {
    const cv = document.getElementById(element.id + "Merge");
    
    if (!cv) {
        console.error(`Canvas not found for ${element.id}`);
        return;
    }
    
    // Wait for video to be ready
    if (element.readyState < 3) {
        console.log(`Waiting for video ${element.id} to load...`);
        element.addEventListener('loadeddata', () => {
            if (element.videoWidth > 0 && element.videoHeight > 0) {
                setupVideoComparison(element, cv);
            }
        });
        return;
    }
    
    setupVideoComparison(element, cv);
}

function resizeAndPlay3(element) {
    const cv = document.getElementById(element.id + "Merge3");
    
    if (!cv) {
        console.error(`Canvas not found for ${element.id}`);
        return;
    }
    
    // Wait for video to be ready
    if (element.readyState < 3) {
        console.log(`Waiting for video ${element.id} to load...`);
        element.addEventListener('loadeddata', () => {
            if (element.videoWidth > 0 && element.videoHeight > 0) {
                setupVideoComparison3(element, cv);
            }
        });
        return;
    }
    
    setupVideoComparison3(element, cv);
}

function setupVideoComparison(element, canvas) {
    canvas.width = element.videoWidth / 2;
    canvas.height = element.videoHeight;
    element.play();
    element.style.height = "0px"; // Hide video without stopping it
    
    console.log(`Setting up 2-way comparison for ${element.id}, canvas size: ${canvas.width}x${canvas.height}`);
    playVids(element.id);
}

function setupVideoComparison3(element, canvas) {
    canvas.width = element.videoWidth / 3;
    canvas.height = element.videoHeight;
    element.play();
    element.style.height = "0px"; // Hide video without stopping it
    
    console.log(`Setting up 3-way comparison for ${element.id}, canvas size: ${canvas.width}x${canvas.height}`);
    playVids3(element.id);
}
