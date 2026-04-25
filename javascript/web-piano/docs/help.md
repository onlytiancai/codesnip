
These are some of the most common things that can be done with ABC. For all the details, see [the full documentation.](https://abcnotation.com/wiki/abc:standard)

### Headers

Each tune needs to start off with a set of headers. See the full documentation for a detailed discussion of what headers are possible. Described below are the most used header elements. Headers always start with a capital letter as the first character on a line and a colon as the second letter.

Header fields that appear before any of the music:

| Field | Typical Value | Meaning |
| --- | --- | --- |
| X: | 1 | Must be the first line of each tune. The value it is given is not used anywhere. |
| T: | Twinkle, Twinkle | This appears as a large title on top of the music. |
| C: | Anonymous | This appears on top of the music on the right side. |
| L: | 1/4 or 1/8 | Which duration is the default. Typically this is either a quarter note or an eighth note. |
| M: | 2/4, 3/4, 4/4, 6/8, C, or C| | A time signature. Typical ones are shown, but there are many that are accepted. |
| Q: | 1/4=120 | The tempo of the piece. Typically the beat is chosen for before the equal sign, and the beats per minute is chosen to the right of the equal sign. |
| K: | D or Dm | The key, starting with a letter from A to G, then possibly having an "m" to indicate a minor key. Note that many other modes and many other parameters are accepted on this line, so see the full documentation. |

```
X:1
T:Twinkle, Twinkle
C:Anonymous
L:1/4
M:4/4
Q:1/4=120
K:D
DDAA|BBA2|]
```


Header fields that are interspersed in the music:

| Field | Typical Value | Meaning |
| --- | --- | --- |
| V: | alto | See the section on voices for more details. |
| P: | Chorus | This will be printed between staves and can be any string you like. |

### Pitches

Pitches are specifed with the note name.

A capital letter is an octave below a lower case letter. If a comma appears after a letter, then the note is an octave below that. If an apostrophe appears after a note, then the note is raised an octave.

Rests are specified with the "z" character. The different shaped rests are depend on the duration. (See the duration section for more details.)

See the following example for specifying pitches:


Accidentals (that are not in the key signature) are specified with an underscore for flat, an equal sign for natural, and a caret for sharp. See the following example:


### Durations

The note durations are specified by a number or fraction after the note. The note length that corresponds to 1, or not specifying a duration is in the "L:" header field. A good strategy for choosing a value for the L: field is to see if there are more eight notes or quarter notes, and set the L: field to that.

Note that "/2" can be abbreviated to "/"

Triplets are indicated by starting the three notes with "(3". Note that the subject of triplets and other duples are complicated, so if you want to do anything more elaborate than that, see the official documentation, linked to above.

The following example shows the way to get the different durations for both values of L:


### Bar Lines

Many different styles of bar lines are possible. First and second endings are also possible. A chart follows with all of the commonly used bar lines.

| Bar | Meaning |
| --- | --- |
| \| | bar line |
| \|\] | thin-thick double bar line |
| \|\| | thin-thin double bar line |
| \[\| | thick-thin double bar line |
| \|: | start of repeated section |
| :\| | end of repeated section |
| :: | start & end of two repeated sections |
| \|1 | start of first ending |
| :\|2 | start of second ending |


### Beaming

Beaming is done whenever possible. So whenever there are more than one note that is shorter than a quarter note, they will be beamed together. If you don't want beaming, then you can break the beam with a space between notes. That is, there is no beaming done over a space.

```
X:1
L:1/8
K:C
A B c d AB cd ABcd|
```


### Ties and Slurs

Ties are between two notes of the same pitch. That means that it sounds as one note. Slurs are between two notes of different pitches. Ties and slurs looks the same but they are handled slightly different by the MIDI and animation routines. A tie is expressed by a hyphen between two notes. A slur is expressed by putting the notes to be slurred in parentheses.

```
X:1
L:1/8
K:C
(Ac) d-d| (Ac eg) ga- a-a|
```

### Accompaniment

Chord symbols can be placed over the staff to indicate what chords the accompanist should play. These chords are played by the MIDI player, too. Chords are created by putting any string inside double quote marks. There is no error checking of these strings. They are just printed as is. Many common patterns are understood by the MIDI player.

```
X:1
L:1/4
K:C
"A"A "Gm7"D "Bb°7"F "F#9"g|]
```


Here's a (partial) list of the chord types that are understood by the playback system:

| Suffix | Meaning |
| --- | --- |
| (none) or M | Major chord |
| maj7 or ∆7 | Major seventh chord (Option + j) |
| 6 | Major sixth chord |
| 7 | Dominant seventh chord |
| + | Augmented chord |
| +7 or aug7 or 7#5 or 7+5 | Augmented seventh chord |
| \- or m | Minor chord |
| \-6 or m6 | Minor sixth chord |
| \-7 or m7 | Minor seventh chord |
| dim or ° | Diminished chord (Option + Shift + 8) |
| dim7 or °7 | Diminished seventh chord |
| ø7 | Half-diminished seventh chord (Option + o) |
| 9 | Ninth chord |
| 11 | Eleventh chord |
| 13 | Thirteenth chord |
| 7b9 | Dominant seventh, flat nine chord |
| 7b5 | Dominant seventh, flat five chord |
| 9#5 or 9+5 | Dominant seventh, sharp five chord |
| sus4 | Sustained four chord |
| 7sus4 | Dominant seventh, sustained four chord |
| m7sus4 | Minor seventh, sustained four chord |

### Decorations

There are many possible extra symbols that can be added. A few of the most popular ones are demonstrated below. See the full documentation if you don't see one you want to use.

```
X:1
L:1/4
K:C
.C !tenuto!D !marcato!E !>!F | ~G HA TB2 |
!pppp!c !ppp!d !pp!e !p!f | !mp!g !mf!a !f!b !ff!a | !fff!g !ffff!f !sfz!e2 |]
```


### Grace Notes

Grace notes are represented by notes inside of curly braces. They are attached to a following note and printed small.

```
X:1
L:1/4
K:C
{g}A3 A{g}AA|{cAGAG}A3 {g}A{d}A{e}A|]
```


### Multiple Notes

There are a couple of ways to represent multiple notes that are playing at the same time. If the notes are from different voices then see the section on voices. If the notes come from the same instrument (like a piano or guitar), they can be formed by putting the notes that should be stacked inside square brackets.

```
X:1
L:1/8
K:C
[CEGc] [C2G2] [CE][DF]|[D2F2][EG][FA] [A4d4]|]
```


### Lyrics

Lyrics can appear under the notes by interweaving a separate line that starts with "w:". That is, create a line of music like you normally do, then have a line that starts with "w:" and put the lyrics there. Each word of the lyrics corresponds to a note in the music.

If that were all you could do with lyrics, that would be pretty limiting. In songs, there is not a one-to-one correspondence between words and notes: words have more than one syllable and each syllable is a note, and sometimes a syllable is stretched over more than one note.

The following chart shows how the following problems are taken care of:

| Character | Meaning |
| --- | --- |
| \- | (hyphen) break between syllables within a word |
| \_ | (underscore) previous syllable is to be held for an extra note |
| \* | one note is skipped (i.e. \* is equivalent to a blank syllable) |
| ~ | appears as a space; aligns multiple words under one note |
| \\- | appears as hyphen; aligns multiple syllables under one note |
| | | advances to the next bar |

```
X:1
L:1/4
K:C
A A A A| A A A A | A A A A |]
w:word syll-a-ble syll-a--ble time__  of~the~day
```

### Voices / Staves

There are many possibilities for using multiple voices. See the full documentation if the following use cases don't fit your needs.

There are three elements involved. There is the "%%staves" directive for specifying which voices go in which staff, the "V:" field in the header for defining the voices, and the "V:" field in the body for specifying which voice a line of music goes with.

It is easy to make mistakes and mis-align your music when using multiple voices. Make sure each voice has the correct number of beats and the same bar lines and repeat marks.

### Piano

A piano score has two staves and two voices: left hand and right hand. There is a brace connecting the two staves. The top staff is treble clef and the bottom staff is bass clef. Here's a simple example of setting that up:

```
X:1
M:4/4
T:Piano
L:1/4
%%staves {(RH) (LH)}
V:RH clef=treble
V:LH clef=bass
K:C
V: RH
ABcd | [ce]2 [ce]2 | [df]2 [df]2| [ce]4:|]
V: LH
A,2 E,2 | A,2 E,2 | D,2 A,2 | A, G, F, E, :|]
```

### Multiple Instrument Score

If all instruments go on separate staves, then the "%%staves" directive is not needed.

Note that you can write the music in concert pitch but have it appear for Bb or Eb instruments by using the "score=\_B" or "score=\_E" option. (You may change the octave that it transposes to by using a lower case "\_b" or "\_e".)

```
X: 1
T: Score
M: 4/4
L: 1/4
V: Cl name="Clarinet" score=_B
V: Vi name="Violin"
V: Tr name="Trombone" clef=bass
K: Emin
V: Cl
EFGA|Bcde|edcB|AGFE|]
V: Vi
EFGA|Bcde|edcB|AGFE|]
V: Tr
E,F,G,A,|B,c,d,e,|e,d,c,B,|A,G,F,E,|]
```

### Four-part Harmony, two staves

Sometimes parts should be on the same staff. In four-part harmony, typically there are two staves and two voices on each staff.

```
X:1
T:Zocharti Loch
C:Louis Lewandowski (1821-1894)
M:C
Q:1/4=76
%%staves (T1 T2) (B1 B2)
V:T1  clef=treble-8  name="Tenore I"   snm="T.I"
V:T2  clef=treble-8  name="Tenore II"  snm="T.II"
V:B1  clef=bass      name="Basso I"    snm="B.I"
V:B2  clef=bass      name="Basso II"   snm="B.II"
K:Gm
%            End of header, start of tune body:
% 1
[V:T1]  B2c2 d2g2  | f6e2      | d2c2 d2e2 | d4 c2z2 |
[V:T2]  (G2A2 B2e2)  | d6c2      | (B2A2 B2)c2 | B4 A2z2 |
[V:B1]       z8      | z2f,,2 g,,2a,,2 | b,,2z2 z2 e,,2  | f,,4 f,,2z2 |
[V:B2]       x8      |     x8    |      x8     |    x8   |
% 5
[V:T1]  (B2c2 d2g2)  | f8        | d3c (d2fe)  | H d6    ||
[V:T2]       z8      |     z8    | B3A (B2c2)  | A6    ||
[V:B1]  (d,,2f,,2 b,,2e,2) | d,8       | g,,3g,,  g,,4     | H^f,,6    ||
[V:B2]       x8      | z2B,,2 c,,2d,,2 | e,,3e,, (d,,2c,,2)  | d,,6    ||
```

### Comments

Anything after a single percent sign (%) on a line is ignored. That means you can freely add comments about your tune to make it easier to read.

A tune ends when the first blank line is encountered, but if you'd like to use a blank line, the next best thing is a line that contains only a percent sign.

If a line starts with two percent signs (%%) what follows is a formatting directive. There are many of these. See the full documentation for more details.

```
X:1
%%leftmargin 100
L:1/4
% This line is ignored
K:C
%
% This is part of a C-scale.
%
cdef|] % This text is ignored, too.
``