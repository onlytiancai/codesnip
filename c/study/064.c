#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_WORD_LENGTH 100
#define MAX_WORDS 10000

typedef struct {
  char word[MAX_WORD_LENGTH];
  int count;
} WordCount;

int compareWordCounts(const void *a, const void *b) {
  const WordCount *wc1 = (WordCount *)a;
  const WordCount *wc2 = (WordCount *)b;
  return wc2->count - wc1->count; // Sort in descending order of count
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: word_counter <filename>\n");
    return 1;
  }

  char *filename = argv[1];
  FILE *file = fopen(filename, "r");
  if (!file) {
    printf("Error: Cannot open file '%s'\n", filename);
    return 1;
  }

  WordCount wordCounts[MAX_WORDS];
  int numWords = 0;

  char line[MAX_WORD_LENGTH];
  while (fgets(line, sizeof(line), file)) {
    char *word = strtok(line, " ");
    while (word) {
      // Convert word to lowercase
      for (int i = 0; word[i]; i++) {
        word[i] = tolower(word[i]);
      }

      // Check if word already exists in the array
      int found = 0;
      for (int i = 0; i < numWords; i++) {
        if (strcmp(wordCounts[i].word, word) == 0) {
          wordCounts[i].count++;
          found = 1;
          break;
        }
      }

      // Add new word to the array if not found
      if (!found) {
        if (numWords == MAX_WORDS) {
          printf("Error: Too many unique words. Increase MAX_WORDS.\n");
          fclose(file);
          return 1;
        }

        strcpy(wordCounts[numWords].word, word);
        wordCounts[numWords].count = 1;
        numWords++;
      }

      word = strtok(NULL, " ");
    }
  }

  fclose(file);

  // Sort word counts in descending order
  qsort(wordCounts, numWords, sizeof(WordCount), compareWordCounts);

  // Print the top 10 most frequent words
  printf("Top 10 Most Frequent Words:\n");
  for (int i = 0; i < 10 && i < numWords; i++) {
    printf("%s: %d\n", wordCounts[i].word, wordCounts[i].count);
  }

  return 0;
}
