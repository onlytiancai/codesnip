let currentIndex = 0, lines=[], audio, divs=[];

function playNext() {
  console.log('play next', currentIndex)
  highlightDiv(currentIndex);
  if (currentIndex < lines.length) {
    const line = lines[currentIndex]
    playAudio(`/stream-mp3?txt=${line}`)
    currentIndex++;
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

document.addEventListener("DOMContentLoaded", function() {
  audio = document.getElementById('audioPlayer');
  audio.addEventListener('ended', playNext);

  fetch('/get_lines?n=0&m=100')
    .then(response => {
      if (!response.ok) {
        throw new Error('Network response was not ok ' + response.statusText);
      }
      response.text().then(text=> {
        showLines(text);
        document.getElementById('play').disabled = false;
      });
    })
    .then(data => {
      console.log(data);
    })
    .catch(error => {
      console.error('There has been a problem with your fetch operation:', error);
    });
});
