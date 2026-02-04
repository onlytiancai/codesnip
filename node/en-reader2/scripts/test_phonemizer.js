import { phonemize } from "phonemizer";

const phonemes = await phonemize("Hello world.");
console.log(phonemes); // ['həlˈəʊ wˈɜːld']