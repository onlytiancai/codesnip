import { phonemize } from "phonemizer";

const phonemes = await phonemize("FLOPPINUX was released in 2021. After four years people find it helpful. Because of that I decided to revisit FLOPPINUX in 2025 and make updated tutorial. This brings bunch of updates like latest kernel and persistent storage.");
console.log(phonemes); // ['həlˈəʊ wˈɜːld']