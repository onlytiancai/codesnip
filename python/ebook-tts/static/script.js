let currentIndex = 0, lines=[], audio, divs=[];
let pageNo = 0, pageSize = 10;

function playNext() {
  console.log('play next', currentIndex)
  highlightDiv(currentIndex);
  if (currentIndex < lines.length) {
    const line = lines[currentIndex]
    playAudio(`/stream-mp3?txt=${line}`)
    currentIndex++;
  } else {
    pageNext();
  }
}

function highlightDiv(index) {
  divs.forEach((div, i) => {
    if (i === index) {
      div.style.backgroundColor = 'yellow';
    } else {
      div.style.backgroundColor = 'white';
    }
  });
}

function showLines(text) {
  const container = document.getElementById('container');
  lines = [];
  divs = [];
  currentIndex = 0;
  container.innerHTML = '';
  text.split('\n').forEach(line => {
    line = line.trim();
    if (!line) return;
    const div = document.createElement('div');
    divs.push(div);
    lines.push(line);
    div.className = 'line';
    div.textContent = line;
    div.addEventListener('click', function() {
      playAudio(`/stream-mp3?txt=${line}`)
    });
    container.appendChild(div);
  });
}



function playAudio(src) {
  audio.src = src;
  audio.play();
}

function getLines(n,m,callback) {
  startLoading()
  fetch(`/get_lines?n=${n}&m=${m}`)
    .then(response => {
      if (!response.ok) {
        throw new Error('Network response was not ok ' + response.statusText);
      }
      response.text().then(text=> {
        showLines(text);
        document.getElementById('play').disabled = false;
        if (callback) callback();
      });
    })
    .then(data => {
      console.log(data);
    })
    .catch(error => {
      console.error('There has been a problem with your fetch operation:', error);
    })
    .finally(() => {
      stopLoading()
    });

}

function pageBack() {
  setPage(pageNo-1, playNext)
}

function pageNext() {
  setPage(pageNo+1, playNext)
}

function setPage(n,callback) {
  pageNo = n; 
  savePageNo(pageNo);
  startLine = pageNo*pageSize;
  document.getElementById('lblPage').innerText= pageNo+1;
  getLines(startLine, startLine+pageSize, callback);
}

function startLoading() {
  document.getElementById('loadingOverlay').style.display = 'block';
}

function stopLoading() {
  document.getElementById('loadingOverlay').style.display = 'none';
}

const KEY_PAGE_NO= 'page-no';
function savePageNo(number) {
    if (typeof number !== 'number') {
        console.error('The value must be a number.');
        return;
    }
    localStorage.setItem(KEY_PAGE_NO, number.toString());
}

function loadPageNo() {
    const value = localStorage.getItem(KEY_PAGE_NO);
    if (value === null) {
        return 0;
    }
    const number = parseFloat(value);
    if (isNaN(number)) {
        return 0;
    }
    return number;
}

document.addEventListener("DOMContentLoaded", function() {
  audio = document.getElementById('audioPlayer');
  audio.addEventListener('ended', playNext);
  pageNo = loadPageNo();
  console.log('load page number', pageNo);
  setPage(pageNo);
});
