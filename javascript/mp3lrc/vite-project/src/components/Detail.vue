<script setup>
import { onMounted, ref, watch } from 'vue'
import { useRoute } from 'vue-router'

const route = useRoute()
let title = ref('');

// lrc (String) - lrc file text
function parseLyric(lrc) {
    // will match "[00:00.00] ooooh yeah!"
    // note: i use named capturing group
    const regex = /^\[(?<time>\d{2}:\d{2}(.\d{2})?)\](?<text>.*)/;

    // split lrc string to individual lines
    const lines = lrc.split("\n");

    const output = [];

    lines.forEach(line => {
        const matches = line.match(regex);

        // if doesn't match, return.
        if (matches == null) return;

        const { time, text } = matches.groups;

        output.push({
            time: parseTime(time),
            text: text.trim()
        });

    });

    // parse formated time
    // "03:24.73" => 204.73 (total time in seconds)
    function parseTime(time) {
        const minsec = time.split(":");

        const min = parseInt(minsec[0]) * 60;
        const sec = parseFloat(minsec[1]);

        return min + sec;
    }

    return output;
}

// lyrics (Array) - output from parseLyric function
// time (Number) - current time from audio player
function syncLyric(lyrics, time) {
    const scores = [];

    lyrics.forEach(lyric => {
        // get the gap or distance or we call it score
        const score = time - lyric.time;

        // we don't want a negative score
        if (score >= 0) scores.push(score);
    });

    if (scores.length == 0) return null;

    // get the smallest value from scores
    const closest = Math.min(...scores);

    // return the index of closest lyric
    return scores.indexOf(closest);
    //https://dev.to/mcanam/javascript-lyric-synchronizer-4i15
}

onMounted(() => {
    console.log(route.query);
    title.value = route.query.title
})

watch(
    () => route.query.title,
    newTitle => {
        title.value = newTitle
    }
)
</script>
<template>
    <header>{{ title }}</header>

    <div class="container">
        <ul class="wordList">
        </ul>
    </div>

    <footer>
        <div class="buttons">
            <select id="sel_subtitle">
                <option value="en">英文</option>
                <option value="cn">中文</option>
                <option value="en_cn">中英</option>
            </select>
        </div>
        <div class="player-container">
            <audio class="player" controls></audio>

        </div>

    </footer>
</template>

<style scoped>
* {
    box-sizing: border-box;
}

html {
    font-family: sans-serif;
    color: #efefef;
}

body {
    width: 100vw;
    height: 100vh;
    margin: 0;
    padding: 0;
    background-color: black;
    display: flex;
    flex-direction: column;
}

header,
footer {
    background-color: #333;
    color: #fff;
    height: 10vh;
    line-height: 10vh;
    padding-left: 10px;
    display: flex;
    font-size: 16px;
}

footer .buttons {
    width: 50px;
}

footer .buttons select {
    border-radius: 0;
}

footer .player-container {
    flex: 1;
    padding: 0 10px;
}

.player {
    height: 20px;
    width: 100%;
    vertical-align: middle;
}

.container {
    flex: 1;
    overflow: hidden;

}

.container ul {
    list-style: none;
    transition: all 0.6s;
    text-align: center;
    padding: 0;
}

.container li {
    color: #666;
    cursor: pointer;
}

.container li p {
    line-height: 30px;
    margin: 0;
}

.container li p.li_cn {
    display: none;
}

.container li:hover {
    color: lightblue;
}

.container li.active {
    color: #fff;
}
</style>