(function($) {
/*

jQuery-turtle
=============

version 2.0.9

jQuery-turtle is a jQuery plugin for turtle graphics.

With jQuery-turtle, every DOM element is a turtle that can be

moved using turtle graphics methods like fd (forward), bk (back),
rt (right turn), and lt (left turn).  The pen function allows
a turtle to draw on a full-document canvas as it moves.

<pre>
$('#turtle').pen('red').rt(90).fd(100).lt(90).bk(50).fadeOut();
</pre>

jQuery-turtle provides:
  * Relative and absolute motion and drawing.
  * Functions to ease basic input, output, and game-making for beginners.
  * Operations on sets of turtles, and turtle motion of arbitrary elements.
  * Accurate collision-testing of turtles with arbitrary convex hulls.
  * Simplified access to CSS3 transforms, jQuery animations, Canvas, and Web Audio.
  * An interactive turtle console in either Javascript or CoffeeScript.

The plugin can also create a learning environment with a default
turtle that is friendly for beginners.  The following is a complete
CoffeeScript program that uses the default turtle to draw a grid of
sixteen colored polygons.

<pre>
eval $.turtle()  # Create the default turtle.

speed 100
for color in [red, gold, green, blue]
  for sides in [3..6]
    pen color
    for x in [1..sides]
      fd 100 / sides
      lt 360 / sides
    pen null
    fd 40
  slide 40, -160
</pre>

[Try an interactive demo (CoffeeScript syntax) here.](
http://davidbau.github.io/jquery-turtle/demo.html)


JQuery Methods for Turtle Movement
----------------------------------

The turtle API is briefly summarized below.  All the following
turtle-oriented methods operate on any jQuery object (including
the default turtle, if used):

<pre>
$(q).fd(100)      // Forward relative motion in local coordinates.
$(q).bk(50)       // Back.
$(q).rt(90)       // Right turn.  Optional second arg is turning radius.
$(q).lt(45)       // Left turn.  Optional second arg is turning radius.
$(q).slide(x, y)  // Move right by x while moving forward by y.
$(q).leap(x, y)   // Like slide, but without drawing.
$(q).moveto({pageX:x,pageY:y} | [x,y])  // Absolute motion on page.
$(q).jumpto({pageX:x,pageY:y} | [x,y])  // Like moveto, without drawing.
$(q).turnto(direction || position)      // Absolute direction adjustment.
$(q).play("ccgg") // Plays notes using ABC notation and waits until done.

// Methods below happen in an instant, but line up in the animation queue.
$(q).home()       // Jumps to the center of the document, with direction 0.
$(q).pen('red')   // Sets a pen style, or 'none' for no drawing.
$(q).pu()         // Pen up - temporarily disables the pen (also pen(false)).
$(q).pd()         // Pen down - starts a new pen path.
$(q).pe()         // Uses the pen 'erase' style.
$(q).fill('gold') // Fills a shape previously outlined using pen('path').
$(q).dot(12)      // Draws a circular dot of diameter 12.  Color second arg.
$(q).label('A')   // Prints an HTML label at the turtle location.
$(q).speed(10)    // Sets turtle animation speed to 10 moves per sec.
$(q).ht()         // Hides the turtle.
$(q).st()         // Shows the turtle.
$(q).wear('blue') // Switches to a blue shell.  Use any image or color.
$(q).scale(1.5)   // Scales turtle size and motion by 150%.
$(q).twist(180)   // Changes which direction is considered "forward".
$(q).mirror(true) // Flips the turtle across its main axis.
$(q).reload()     // Reloads the turtle's image (restarting animated gifs)
$(q).done(fn)     // Like $(q).promise().done(fn). Calls after all animation.
$(q).plan(fn)     // Like each, but this is set to $(elt) instead of elt,
                  // and the callback fn can insert into the animation queue.

// Methods below this line do not queue for animation.
$(q).getxy()      // Local (center-y-up [x, y]) coordinates of the turtle.
$(q).pagexy()     // Page (topleft-y-down {pageX:x, pageY:y}) coordinates.
$(q).direction([p]) // The turtles absolute direction (or direction towards p).
$(q).distance(p)  // Distance to p in page coordinates.
$(q).shown()      // Shorthand for is(":visible")
$(q).hidden()     // Shorthand for !is(":visible")
$(q).touches(y)   // Collision tests elements (uses turtleHull if present).
$(q).inside(y)// Containment collision test.
$(q).nearest(pos) // Filters to item (or items if tied) nearest pos.
$(q).within(d, t) // Filters to items with centers within d of t.pagexy().
$(q).notwithin()  // The negation of within.
$(q).cell(y, x)   // Selects the yth row and xth column cell in a table.
</pre>


Speed and Turtle Animation
--------------------------

When the speed of a turtle is nonzero, the first nine movement
functions animate at that speed (in moves per second), and the
remaining mutators also participate in the animation queue.  The
default turtle speed is a leisurely one move per second (as
appropriate for the creature), but you may soon discover the
desire to set speed higher.

Setting the turtle speed to Infinity will make its movement synchronous,
which makes the synchronous distance, direction, and hit-testing useful
for realtime game-making.

Pen and Fill Styles
-------------------

The turtle pen respects canvas styling: any valid strokeStyle is
accepted; and also using a css-like syntax, lineWidth, lineCap,
lineJoin, miterLimit, and fillStyle can be specified, e.g.,
pen('red;lineWidth:5;lineCap:square').  The same syntax applies for
styling dot and fill (except that the default interpretation for the
first value is fillStyle instead of strokeStyle).

The fill method is used by tracing an invisible path using the
pen('path') style, and then calling the fill method.  Disconnected
paths can be created using pu() and pd().

Conventions for Musical Notes
-----------------------------

The play method plays a sequence of notes specified using a subset of
standard ABC notation.  Capital C denotes middle C, and lowercase c is
an octave higher.  Pitches and durations can be altered with commas,
apostrophes, carets, underscores, digits, and slashes as in the
standard.  Enclosing letters in square brackets represents a chord,
and z represents a rest.  The default tempo is 120, but can be changed
by passing a options object as the first parameter setting tempo, e.g.,
{ tempo: 200 }.

The turtle's motion will pause while it is playing notes. A single
tone can be played immediately (without participating in the
turtle animation queue) by using the "tone" method.

Planning Logic in the Animation Queue
-------------------------------------

The plan method can be used to queue logic (including synchronous
tests or actions) by running a function in the animation queue.  Unlike
jquery queue(), plan arranges things so that if further animations
are queued by the callback function, they are inserted (in natural
recursive functional execution order) instead of being appended.

Turnto and Absolute Bearings
----------------------------

The turnto method can turn to an absolute direction (if called with a
single numeric argument) or towards an absolute position on the
screen.  The methods moveto and turnto accept either page or
graphing coordinates.

Moveto and Two Flavors of Cartesian Coordinates
-----------------------------------------------

Graphing coordinates are measured upwards and rightwards from the
center of the page, and they are specified as bare numeric x, y
arguments or [x, y] pairs as returned from getxy().

Page coordinates are specified by an object with pageX and pageY
properties, or with a pagexy() method that will return such an object.
That includes, usefullly, mouse events and turtle objects.  Page
coordinates are measured downward from the top-left corner of the
page to the center (or transform-origin) of the given object.

Hit Testing
-----------

The hit-testing functions touches() and inside() will test for
collisions using the convex hulls of the objects in question.
The hull of an element defaults to the bounding box of the element
(as transformed) but can be overridden by the turtleHull CSS property,
if present.  The default turtle is given a turtle-shaped hull.

The touches() function can also test for collisions with a color
on the canvas - use touches('red'), for example, or for collsisions
with any nontransparent color, use touches('color').

Turtle Teaching Environment
---------------------------

A default turtle together with an interactive console are created by
calling eval($.turtle()).  That call exposes all the turtle methods
such as (fd, rt, getxy, etc) as global functions operating on the default
turtle.  It will also set up a number of other global symbols to provide
beginners with a simplified programming environment.

In detail, after eval($.turtle()):
  * An &lt;img id="turtle"&gt; is created if #turtle doesn't already exist.
  * An eval debugging panel (see.js) is shown at the bottom of the screen.
  * Turtle methods on the default turtle are packaged as globals, e.g., fd(10).
  * Every #id element is turned into a global variable: window.id = $('#id').
  * Default turtle animation is set to 1 move per sec so steps can be seen.
  * Global event listeners are created to update global event variables.
  * Methods of $.turtle.* (enumerated below) are exposed as global functions.
  * String constants are defined for the 140 named CSS colors.

Beyond the functions to control the default turtle, the globals added by
$.turtle() are as follows:

<pre>
lastclick             // Event object of the last click event in the doc.
lastdblclick          // The last double-click event.
lastmousemove         // The last mousemove event.
lastmouseup           // The last mouseup event.
lastmousedown         // The last mousedown event.
keydown               // The last keydown event.
keyup                 // The last keyup event.
keypress              // The last keypress event.
hatch([n,] [img])     // Creates and returns n turtles with the given img.
cs()                  // Clears the screen, both the canvas and the body text.
cg()                  // Clears the graphics canvas without clearing the text.
ct()                  // Clears the text without clearing the canvas.
defaultspeed(mps)     // Sets $.fx.speeds.turtle to 1000 / mps.
timer(secs, fn)       // Calls back fn once after secs seconds.
tick([perSec,] fn)    // Repeatedly calls fn at the given rate (null clears).
done(fn)              // Calls back fn after all turtle animation is complete.
random(n)             // Returns a random number [0..n-1].
random(n,m)           // Returns a random number [n..m-1].
random(list)          // Returns a random element of the list.
random('normal')      // Returns a gaussian random (mean 0 stdev 1).
random('uniform')     // Returns a uniform random [0...1).
random('position')    // Returns a random {pageX:x, pageY:y} coordinate.
random('color')       // Returns a random hsl(*, 100%, 50%) color.
random('gray')        // Returns a random hsl(0, 0, *) gray.
remove()              // Removes default turtle and its globals (fd, etc).
see(a, b, c...)       // Logs tree-expandable data into debugging panel.
write(html)           // Appends html into the document body.
type(plaintext)       // Appends preformatted text into a pre in the document.
read([label,] fn)     // Makes a one-time input field, calls fn after entry.
readnum([label,] fn)  // Like read, but restricted to numeric input.
readstr([label,] fn)  // Like read, but never converts input to a number.
button([label,] fn)   // Makes a clickable button, calls fn when clicked.
menu(choices, fn)     // Makes a clickable choice, calls fn when chosen.
table(m, n)           // Outputs a table with m rows and n columns.
play('[DFG][EGc]')    // Plays musical notes.
send(m, arg)          // Sends an async message to be received by recv(m, fn).
recv(m, fn)           // Calls fn once to receive one message sent by send.
</pre>

Here is another CoffeeScript example that demonstrates some of
the functions:

<pre>
eval $.turtle()  # Create the default turtle and global functions.

speed Infinity
write "Catch blue before red gets you."
bk 100
r = new Turtle red
b = new Turtle blue
tick 10, ->
  turnto lastmousemove
  fd 6
  r.turnto turtle
  r.fd 4
  b.turnto direction b
  b.fd 3
  if b.touches(turtle)
    write "You win!"
    tick off
  else if r.touches(turtle)
    write "Red got you!"
    tick off
  else if not b.inside(document)
    write "Blue got away!"
    tick off
</pre>

The turtle teaching environment is designed to work well with either
Javascript or CoffeeScript.

JQuery CSS Hooks for Turtle Geometry
------------------------------------

Underlying turtle motion are turtle-oriented 2d transform jQuery cssHooks,
with animation support on all motion:

<pre>
$(q).css('turtleSpeed', '10');         // speed in moves per second.
$(q).css('turtleEasing', 'linear');    // animation easing, defaults to swing.
$(q).css('turtlePosition', '30 40');   // position in local coordinates.
$(q).css('turtlePositionX', '30px');   // x component.
$(q).css('turtlePositionY', '40px');   // y component.
$(q).css('turtleRotation', '90deg');   // rotation in degrees.
$(q).css('turtleScale', '2');          // double the size of any element.
$(q).css('turtleScaleX', '2');         // x stretch after twist.
$(q).css('turtleScaleY', '2');         // y stretch after twist.
$(q).css('turtleTwist', '45deg');      // turn before stretching.
$(q).css('turtleForward', '50px');     // position in direction of rotation.
$(q).css('turtleTurningRadius, '50px');// arc turning radius for rotation.
$(q).css('turtlePenStyle', 'red');     // or 'red lineWidth 2px' etc.
$(q).css('turtlePenDown', 'up');       // default 'down' to draw with pen.
$(q).css('turtleHull', '5 0 0 5 0 -5');// fine-tune shape for collisions.
$(q).css('turtleTimbre', 'square');    // quality of the sound.
$(q).css('turtleVolume', '0.3');       // volume of the sound.
</pre>

Arbitrary 2d transforms are supported, including transforms of elements
nested within other elements that have css transforms. For example, arc
paths of a turtle within a skewed div will transform to the proper elliptical
arc.  Note that while turtle motion is transformed, lines and dots are not:
for example, dots are always circular.  To get transformed circles, trace
out an arc.

Transforms on the turtle itself are used to infer the turtle position,
direction, and rendering of the sprite.  ScaleY stretches the turtle
sprite in the direction of movement also stretches distances for
motion in all directions.  ScaleX stretches the turtle sprite perpendicular
to the direction of motion and also stretches line and dot widths for
drawing.

A canvas is supported for drawing, but only created when the pen is
used; pen styles include canvas style properties such as lineWidth
and lineCap.

A convex hull polygon can be set to be used by the collision detection
and hit-testing functions (inside, touches).  The turtleHull is a list
of (unrotated) x-y coordinates relative to the object's transformOrigin.
If set to 'auto' (the default) the hull is just the bounding box for the
element.

License (MIT)
-------------

Copyright (c) 2013 David Bau

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

*/

//////////////////////////////////////////////////////////////////////////
// PREREQUISTIES
// Establish support for transforms in this browser.
//////////////////////////////////////////////////////////////////////////

var undefined = void 0,
    global = this,
    __hasProp = {}.hasOwnProperty,
    rootjQuery = jQuery(function() {}),
    interrupted = false,
    async_pending = 0,
    global_plan_depth = 0,
    global_plan_queue = [];

function nonrecursive_dequeue(elem, qname) {
  if (global_plan_depth > 5) {
    global_plan_queue.push({elem: elem, qname: qname});
  } else {
    global_plan_depth += 1;
    $.dequeue(elem, qname);
    while (global_plan_queue.length > 0) {
      var task = global_plan_queue.shift();
      $.dequeue(task.elem, task.qname);
      checkForHungLoop()
    }
    global_plan_depth -= 1;
  }
}

function __extends(child, parent) {
  for (var key in parent) {
    if (__hasProp.call(parent, key)) child[key] = parent[key];
  }
  function ctor() { this.constructor = child; }
  ctor.prototype = parent.prototype;
  child.prototype = new ctor();
  child.__super__ = parent.prototype;
  return child;
};

if (!$.cssHooks) {
  throw("jQuery 1.4.3+ is needed for jQuery-turtle to work");
}

// Determine the name of the 'transform' css property.
function styleSupport(prop) {
  var vendorProp, supportedProp,
      capProp = prop.charAt(0).toUpperCase() + prop.slice(1),
      prefixes = [ "Moz", "Webkit", "O", "ms" ],
      div = document.createElement("div");
  if (prop in div.style) {
    supportedProp = prop;
  } else {
    for (var i = 0; i < prefixes.length; i++) {
      vendorProp = prefixes[i] + capProp;
      if (vendorProp in div.style) {
        supportedProp = vendorProp;
        break;
      }
    }
  }
  div = null;
  $.support[prop] = supportedProp;
  return supportedProp;
}
function hasGetBoundingClientRect() {
  var div = document.createElement("div"),
      result = ('getBoundingClientRect' in div);
  div = null;
  return result;
}
var transform = styleSupport("transform"),
    transformOrigin = styleSupport("transformOrigin");

if (!transform || !hasGetBoundingClientRect()) {
  // Need transforms and boundingClientRects to support turtle methods.
  return;
}

// An options string looks like a (simplified) CSS properties string,
// of the form prop:value;prop:value; etc.  If defaultProp is supplied
// then the string can begin with "value" (i.e., value1;prop:value2)
// and that first value will be interpreted as defaultProp:value1.
// Some rudimentary quoting can be done, e.g., value:"prop", etc.
function parseOptionString(str, defaultProp) {
  if (typeof(str) != 'string') {
    if (str == null) {
      return {};
    }
    if ($.isPlainObject(str)) {
      return str;
    }
    str = '' + str;
  }
  // Each token is an identifier, a quoted or parenthesized string,
  // a run of whitespace, or any other non-matching character.
  var token = str.match(/[-a-zA-Z_][-\w]*|"[^"]*"|'[^']'|\([^()]*\)|\s+|./g),
      result = {}, j, t, key = null, value, arg,
      seencolon = false, vlist = [], firstval = true;

  // While parsing, commitvalue() validates and unquotes a prop:value
  // pair and commits it to the result.
  function commitvalue() {
    // Trim whitespace
    while (vlist.length && /^\s/.test(vlist[vlist.length - 1])) { vlist.pop(); }
    while (vlist.length && /^\s/.test(vlist[0])) { vlist.shift(); }
    if (vlist.length == 1 && (
          /^".*"$/.test(vlist[0]) || /^'.*'$/.test(vlist[0]))) {
      // Unquote quoted string.
      value = vlist[0].substr(1, vlist[0].length - 2);
    } else if (vlist.length == 2 && vlist[0] == 'url' &&
        /^(.*)$/.test(vlist[1])) {
      // Remove url(....) from around a string.
      value = vlist[1].substr(1, vlist[1].length - 2);
    } else {
      // Form the string for the value.
      arg = vlist.join('');
      // Convert the value to a number if it looks like a number.
      if (arg == "") {
        value = arg;
      } else if (isNaN(arg)) {
        value = arg;
      } else {
        value = Number(arg);
      }
    }
    // Deal with a keyless first value.
    if (!seencolon && firstval && defaultProp && vlist.length) {
      // value will already have been formed.
      key = defaultProp;
    }
    if (key) {
      result[key] = value;
    }
  }
  // Now the parsing: just iterate through all the tokens.
  for (j = 0; j < token.length; ++j) {
    t = token[j];
    if (!seencolon) {
      // Before a colon, remember the first identifier as the key.
      if (!key && /^[a-zA-Z_-]/.test(t)) {
        key = t;
      }
      // And also look for the colon.
      if (t == ':') {
        seencolon = true;
        vlist.length = 0;
        continue;
      }
    }
    if (t == ';') {
      // When a semicolon is seen, form the value and save it.
      commitvalue();
      // Then reset the parsing state.
      key = null;
      vlist.length = 0;
      seencolon = false;
      firstval = false;
      continue;
    }
    // Accumulate all tokens into the vlist.
    vlist.push(t);
  }
  commitvalue();
  return result;
}
// Prints a map of options as a parsable string.
// The inverse of parseOptionString.
function printOptionAsString(obj) {
  var result = [];
  function quoted(s) {
    if (/[\s;]/.test(s)) {
      if (s.indexOf('"') < 0) {
        return '"' + s + '"';
      }
      return "'" + s + "'";
    }
    return s;
  }
  for (var k in obj) if (obj.hasOwnProperty(k)) {
    result.push(k + ':' + quoted(obj[k]) + ';');
  }
  return result.join(' ');
}

//////////////////////////////////////////////////////////////////////////
// MATH
// 2d matrix support functions.
//////////////////////////////////////////////////////////////////////////

function identity(x) { return x; }

// Handles both 2x2 and 2x3 matrices.
function matrixVectorProduct(a, v) {
  var r = [a[0] * v[0] + a[2] * v[1], a[1] * v[0] + a[3] * v[1]];
  if (a.length == 6) {
    r[0] += a[4];
    r[1] += a[5];
  }
  return r;
}

// Multiplies 2x2 or 2x3 matrices.
function matrixProduct(a, b) {
  var r = [
    a[0] * b[0] + a[2] * b[1],
    a[1] * b[0] + a[3] * b[1],
    a[0] * b[2] + a[2] * b[3],
    a[1] * b[2] + a[3] * b[3]
  ];
  var along = (a.length == 6);
  if (b.length == 6) {
    r.push(a[0] * b[4] + a[2] * b[5] + (along ? a[4] : 0));
    r.push(a[1] * b[4] + a[3] * b[5] + (along ? a[5] : 0));
  } else if (along) {
    r.push(a[4]);
    r.push(a[5]);
  }
  return r;
}

function nonzero(e) {
  // Consider zero any deviations less than one in a trillion.
  return Math.abs(e) > 1e-12;
}

function isone2x2(a) {
  return !nonzero(a[1]) && !nonzero(a[2]) &&
      !nonzero(1 - a[0]) && !nonzero(1 - a[3]);
}

function inverse2x2(a) {
  if (isone2x2(a)) { return [1, 0, 0, 1]; }
  var d = decomposeSVD(a);
  // Degenerate matrices have no inverse.
  if (!nonzero(d[2])) return null;
  return matrixProduct(
      rotation(-(d[3])), matrixProduct(
      scale(1/d[1], 1/d[2]),
      rotation(-(d[0]))));
}

// By convention, a 2x3 transformation matrix has a 2x2 transform
// in the first four slots and a 1x2 translation in the last two slots.
// The array [a, b, c, d, e, f] is shorthand for the following 3x3
// matrix where the upper-left 2x2 is an in-place transform, and the
// upper-right vector [e, f] is the translation.
//
//   [a c e]   The inverse of this 3x3 matrix can be formed by
//   [b d f]   figuring the inverse of the 2x2 upper-left corner
//   [0 0 1]   (call that Ai), and then negating the upper-right vector
//             (call that -t) and then forming Ai * (-t).
//
//   [ A  t]   (if Ai is the inverse of A)   [  Ai  Ai*(-t) ]
//   [0 0 1]          --- invert --->        [ 0 0    1     ]
//
// The result is of the same form, and can be represented as an array
// of six numbers.
function inverse2x3(a) {
  var ai = inverse2x2(a);
  if (a.length == 4) return ai;
  var nait = matrixVectorProduct(ai, [-a[4], -a[5]]);
  ai.push(nait[0]);
  ai.push(nait[1]);
  return ai;
}

function rotation(theta) {
  var c = Math.cos(theta),
      s = Math.sin(theta);
  return [c, s, -s, c];
}

function scale(sx, sy) {
  if (arguments.length == 1) { sx = sy; }
  return [sx, 0, 0, sy];
}

function addVector(v, a) {
  return [v[0] + a[0], v[1] + a[1]];
}

function subtractVector(v, s) {
  return [v[0] - s[0], v[1] - s[1]];
}

function scaleVector(v, s) {
  return [v[0] * s, v[1] * s];
}

function translatedMVP(m, v, origin) {
  return addVector(matrixVectorProduct(m, subtractVector(v, origin)), origin);
}

// decomposeSVD:
//
// Decomposes an arbitrary 2d matrix into a rotation, an X-Y scaling,
// and a prescaling rotation (which we call a "twist").  The prescaling
// rotation is only nonzero when there is some skew (i.e, a stretch that
// does not preserve rectilinear angles in the source).
//
// This decomposition is stable, which means that the product of
// the three components is always within near machine precision
// (about ~1e-15) of the original matrix.
//
// Input:  [m11, m21, m12, m22] in column-first order.
// Output: [rotation, scalex, scaley, twist] with rotations in radians.
//
// The decomposition is the unique 2d SVD permuted to fit the contraints:
//  * twist is between +- pi/4
//  * rotation is between +- pi/2
//  * scalex + scaley >= 0.
function decomposeSVD(m) {
  var // Compute M*M
      mtm0 = m[0] * m[0] + m[1] * m[1],
      mtm12 = m[0] * m[2] + m[1] * m[3],
      mtm3 = m[2] * m[2] + m[3] * m[3],
      // Compute right-side rotation.
      phi = -0.5 * Math.atan2(mtm12 * 2, mtm0 - mtm3),
      v0 = Math.cos(phi),
      v1 = Math.sin(phi),  // [v0 v1 -v1 v0]
      // Compute left-side rotation.
      mvt0 = (m[0] * v0 - m[2] * v1),
      mvt1 = (m[1] * v0 - m[3] * v1),
      theta = Math.atan2(mvt1, mvt0),
      u0 = Math.cos(theta),
      u1 = Math.sin(theta),  // [u0 u1 -u1 u0]
      // Compute the singular values.  Notice by computing in this way,
      // the sign is pushed into the smaller singular value.
      sv2c = (m[1] * v1 + m[3] * v0) * u0 - (m[0] * v1 + m[2] * v0) * u1,
      sv1c = (m[0] * v0 - m[2] * v1) * u0 + (m[1] * v0 - m[3] * v1) * u1,
      sv1, sv2;
  // Put phi between -pi/4 and pi/4.
  if (phi < -Math.PI / 4) {
    phi += Math.PI / 2;
    sv2 = sv1c;
    sv1 = sv2c;
    theta -= Math.PI / 2;
  } else {
    sv1 = sv1c;
    sv2 = sv2c;
  }
  // Put theta between -pi and pi.
  if (theta > Math.PI) { theta -= 2 * Math.PI; }
  return [theta, sv1, sv2, phi];
}

// approxBezierUnitArc:
// Returns three bezier curve control points that approximate
// a a unit circle arc from angle a1 to a2 (not including the
// beginning point, which would just be at cos(a1), sin(a1)).
// For a discussion and derivation of this formula,
// google [riskus approximating circular arcs]
function approxBezierUnitArc(a1, a2) {
  var a = (a2 - a1) / 2,
      x4 = Math.cos(a),
      y4 = Math.sin(a),
      x1 = x4,
      y1 = -y4,
      q2 = 1 + x1 * x4 + y1 * y4,
      d = (x1 * y4 - y1 * x4),
      k2 = d && (4/3 * (Math.sqrt(2 * q2) - q2) / d),
      x2 = x1 - k2 * y1,
      y2 = y1 + k2 * x1,
      x3 = x2,
      y3 = -y2,
      ar = a + a1,
      car = Math.cos(ar),
      sar = Math.sin(ar);
  return [
     [x2 * car - y2 * sar, x2 * sar + y2 * car],
     [x3 * car - y3 * sar, x3 * sar + y3 * car],
     [Math.cos(a2), Math.sin(a2)]
  ];
}

//////////////////////////////////////////////////////////////////////////
// CSS TRANSFORMS
// Basic manipulation of 2d CSS transforms.
//////////////////////////////////////////////////////////////////////////

function getElementTranslation(elem) {
  var ts = readTurtleTransform(elem, false);
  if (ts) { return [ts.tx, ts.ty]; }
  var m = readTransformMatrix(elem);
  if (m) { return [m[4], m[5]]; }
  return [0, 0];
}

// Reads out the 2x3 transform matrix of the given element.
function readTransformMatrix(elem) {
  var ts = (global.getComputedStyle ?
      global.getComputedStyle(elem)[transform] :
      $.css(elem, 'transform'));
  if (!ts || ts === 'none') {
    return null;
  }
  // Quick exit on the explicit matrix() case:
  var e =/^matrix\(([\-+.\de]+),\s*([\-+.\de]+),\s*([\-+.\de]+),\s*([\-+.\de]+),\s*([\-+.\de]+)(?:px)?,\s*([\-+.\de]+)(?:px)?\)$/.exec(ts);
  if (e) {
    return [parseFloat(e[1]), parseFloat(e[2]), parseFloat(e[3]),
            parseFloat(e[4]), parseFloat(e[5]), parseFloat(e[6])];
  }
  // Interpret the transform string.
  return transformStyleAsMatrix(ts);
}

// Reads out the css transformOrigin property, if present.
function readTransformOrigin(elem, wh) {
  var hidden = ($.css(elem, 'display') === 'none'),
      swapout, old, name;
  if (hidden) {
    // IE GetComputedStyle doesn't give pixel values for transformOrigin
    // unless the element is unhidden.
    swapout = { position: "absolute", visibility: "hidden", display: "block" };
    old = {};
    for (name in swapout) {
      old[name] = elem.style[name];
      elem.style[name] = swapout[name];
    }
  }
  var gcs = (global.getComputedStyle ?  global.getComputedStyle(elem) : null);
  if (hidden) {
    for (name in swapout) {
      elem.style[name] = old[name];
    }
  }
  var origin = (gcs && gcs[transformOrigin] || $.css(elem, 'transformOrigin'));
  if (origin && origin.indexOf('%') < 0) {
    return $.map(origin.split(' '), parseFloat);
  }
  if (wh) {
    return [wh[0] / 2, wh[1] / 2];
  }
  var sel = $(elem);
  return [sel.width() / 2, sel.height() / 2];
}

// Composes all the 2x2 transforms up to the top.
function totalTransform2x2(elem) {
  var result = [1, 0, 0, 1], t;
  while (elem !== null) {
    t = readTransformMatrix(elem);
    if (t && !isone2x2(t)) {
      result = matrixProduct(t, result);
    }
    elem = elem.parentElement;
  }
  return result.slice(0, 4);
}

// Applies the css 2d transforms specification.
function transformStyleAsMatrix(transformStyle) {
  // Deal with arbitrary transforms:
  var result = [1, 0, 0, 1], ops = [], args = [],
      pat = /(?:^\s*|)(\w*)\s*\(([^)]*)\)\s*/g,
      unknown = transformStyle.replace(pat, function(m) {
        ops.push(m[1].toLowerCase());
        args.push($.map(m[2].split(','), function(s) {
          var v = s.trim().toLowerCase();
          return {
            num: parseFloat(v),
            unit: v.replace(/^[+-.\de]*/, '')
          };
        }));
        return '';
      });
  if (unknown) { return null; }
  for (var index = ops.length - 1; index >= 0; --index) {
    var m = null, a, c, s, t;
    var op = ops[index];
    var arg = args[index];
    if (op == 'matrix') {
      if (arg.length >= 6) {
        m = [arg[0].num, arg[1].num, arg[2].num, arg[3].num,
             arg[4].num, arg[5].num];
      }
    } else if (op == 'rotate') {
      if (arg.length == 1) {
        a = convertToRadians(arg[0]);
        c = Math.cos(a);
        s = Math.sin(a);
        m = [c, -s, c, s];
      }
    } else if (op == 'translate' || op == 'translatex' || op == 'translatey') {
      var tx = 0, ty = 0;
      if (arg.length >= 1) {
        if (arg[0].unit && arg[0].unit != 'px') { return null; } // non-pixels
        if (op == 'translate' || op == 'translatex') { tx = arg[0].num; }
        else if (op == 'translatey') { ty = arg[0].num; }
        if (op == 'translate' && arg.length >= 2) {
          if (arg[1].unit && arg[1].unit != 'px') { return null; }
          ty = arg[1].num;
        }
        m = [0, 0, 0, 0, tx, ty];
      }
    } else if (op == 'scale' || op == 'scalex' || op == 'scaley') {
      var sx = 1, sy = 1;
      if (arg.length >= 1) {
        if (op == 'scale' || op == 'scalex') { sx = arg[0].num; }
        else if (op == 'scaley') { sy = arg[0].num; }
        if (op == 'scale' && arg.length >= 2) { sy = arg[1].num; }
        m = [sx, 0, 0, sy, 0, 0];
      }
    } else if (op == 'skew' || op == 'skewx' || op == 'skewy') {
      var kx = 0, ky = 0;
      if (arg.length >= 1) {
        if (op == 'skew' || op == 'skewx') {
          kx = Math.tan(convertToRadians(arg[0]));
        } else if (op == 'skewy') {
          ky = Math.tan(convertToRadians(arg[0]));
        }
        if (op == 'skew' && arg.length >= 2) {
          ky = Math.tan(convertToRadians(arg[0]));
        }
        m = [1, ky, kx, 1, 0, 0];
      }
    } else {
      // Unrecgonized transformation.
      return null;
    }
    result = matrixProduct(result, m);
  }
  return result;
}

//////////////////////////////////////////////////////////////////////////
// ABSOLUTE PAGE POSITIONING
// Dealing with the element origin, rectangle, and direction on the page,
// taking into account nested parent transforms.
//////////////////////////////////////////////////////////////////////////

function limitMovement(start, target, limit) {
  if (limit <= 0) return start;
  var distx = target.pageX - start.pageX,
      disty = target.pageY - start.pageY,
      dist2 = distx * distx + disty * disty;
  if (limit * limit >= dist2) {
    return target;
  }
  var frac = limit / Math.sqrt(dist2);
  return {
    pageX: start.pageX + frac * distx,
    pageY: start.pageY + frac * disty
  };
}

function limitRotation(start, target, limit) {
  if (limit <= 0) { target = start; }
  else if (limit < 180) {
    var delta = normalizeRotation(target - start);
    if (delta > limit) { target = start + limit; }
    else if (delta < -limit) { target = start - limit; }
  }
  return normalizeRotation(target);
}

function getRoundedCenterLTWH(x0, y0, w, h) {
  return { pageX: Math.floor(x0 + w / 2), pageY: Math.floor(y0 + h / 2) };
}

function getStraightRectLTWH(x0, y0, w, h) {
  var x1 = x0 + w, y1 = y0 + h;
  return [
    { pageX: x0, pageY: y0 },
    { pageX: x0, pageY: y1 },
    { pageX: x1, pageY: y1 },
    { pageX: x1, pageY: y0 }
  ];
}

function cleanedStyle(trans) {
  // Work around FF bug: the browser generates CSS transforms with nums
  // with exponents like 1e-6px that are not allowed by the CSS spec.
  // And yet it doesn't accept them when set back into the style object.
  // So $.swap doesn't work in these cases.  Therefore, we have a cleanedSwap
  // that cleans these numbers before setting them back.
  if (!/e[\-+]/.exec(trans)) {
    return trans;
  }
  var result = trans.replace(/(?:\d+(?:\.\d*)?|\.\d+)e[\-+]\d+/g, function(e) {
    return cssNum(parseFloat(e)); });
  return result;
}

// Returns the turtle's origin (the absolute location of its pen and
// center of rotation when no transforms are applied) in page coordinates.
function getTurtleOrigin(elem, inverseParent, extra) {
  var state = $.data(elem, 'turtleData');
  if (state && state.quickhomeorigin && state.down && state.style && !extra
      && elem.classList && elem.classList.contains('turtle')) {
    return state.quickhomeorigin;
  }
  var hidden = ($.css(elem, 'display') === 'none'),
      swapout = hidden ?
        { position: "absolute", visibility: "hidden", display: "block" } : {},
      substTransform = swapout[transform] = (inverseParent ? 'matrix(' +
          $.map(inverseParent, cssNum).join(', ') + ', 0, 0)' : 'none'),
      old = {}, name, gbcr, transformOrigin;
  for (name in swapout) {
    old[name] = elem.style[name];
    elem.style[name] = swapout[name];
  }
  gbcr = getPageGbcr(elem);
  transformOrigin = readTransformOrigin(elem, [gbcr.width, gbcr.height]);
  for (name in swapout) {
    elem.style[name] = cleanedStyle(old[name]);
  }
  if (extra) {
    extra.gbcr = gbcr;
    extra.localorigin = transformOrigin;
  }
  var result = addVector([gbcr.left, gbcr.top], transformOrigin);
  if (state && state.down && state.style) {
    state.quickhomeorigin = result;
  }
  return result;
}

function wh() {
  // Quirks-mode compatible window height.
  return global.innerHeight || $(global).height();
}

function ww() {
  // Quirks-mode compatible window width.
  return global.innerWidth || $(global).width();
}

function dh() {
  return document.body ? $(document).height() : document.height;
}

function dw() {
  return document.body ? $(document).width() : document.width;
}

function invisible(elem) {
  return elem.offsetHeight <= 0 && elem.offsetWidth <= 0;
}

function makeGbcrLTWH(left, top, width, height) {
  return {
    left: left, top: top, right: left + width, bottom: top + height,
    width: width, height: height
  };
}

function getPageGbcr(elem) {
  if (isPageCoordinate(elem)) {
    return makeGbcrLTWH(elem.pageX, elem.pageY, 0, 0);
  } else if ($.isWindow(elem)) {
    return makeGbcrLTWH(
        $(global).scrollLeft(), $(global).scrollTop(), ww(), wh());
  } else if (elem.nodeType === 9) {
    return makeGbcrLTWH(0, 0, dw(), dh());
  } else if (!('getBoundingClientRect' in elem)) {
    return makeGbcrLTWH(0, 0, 0, 0);
  }
  return readPageGbcr.apply(elem);
}

function isGbcrOutside(center, d2, gbcr) {
  var dy = Math.max(0,
           Math.max(gbcr.top - center.pageY, center.pageY - gbcr.bottom)),
      dx = Math.max(0,
           Math.max(gbcr.left - center.pageX, center.pageX - gbcr.right));
  return dx * dx + dy * dy > d2;
}

function isGbcrInside(center, d2, gbcr) {
  var dy = Math.max(gbcr.bottom - center.pageY, center.pageY - gbcr.top),
      dx = Math.max(gbcr.right - center.pageX, center.pageX - gbcr.left);
  return dx * dx + dy * dy < d2;
}

function isDisjointGbcr(gbcr0, gbcr1) {
  return (gbcr1.right < gbcr0.left || gbcr0.right < gbcr1.left ||
          gbcr1.bottom < gbcr0.top || gbcr0.bottom < gbcr1.top);
}

function gbcrEncloses(gbcr0, gbcr1) {
  return (gbcr1.top >= gbcr0.top && gbcr1.bottom <= gbcr0.bottom &&
          gbcr1.left >= gbcr0.left && gbcr1.right <= gbcr0.right);
}

function polyMatchesGbcr(poly, gbcr) {
  return (poly.length === 4 &&
      poly[0].pageX === gbcr.left && poly[0].pageY === gbcr.top &&
      poly[1].pageX === gbcr.left && poly[1].pageY === gbcr.bottom &&
      poly[2].pageX === gbcr.right && poly[2].pageY === gbcr.bottom &&
      poly[3].pageX === gbcr.right && poly[3].pageY === gbcr.top);
}

function readPageGbcr() {
  var raw = this.getBoundingClientRect();
  return {
    top: raw.top + global.pageYOffset,
    bottom: raw.bottom + global.pageYOffset,
    left: raw.left + global.pageXOffset,
    right: raw.right + global.pageXOffset,
    width: raw.width,
    height: raw.height
  };
}

// Temporarily eliminate transform (but reverse parent distortions)
// to get origin position; then calculate displacement needed to move
// turtle to target coordinates (again reversing parent distortions
// if possible).
function computeTargetAsTurtlePosition(elem, target, limit, localx, localy) {
  var totalParentTransform = totalTransform2x2(elem.parentElement),
      inverseParent = inverse2x2(totalParentTransform),
      origin = getTurtleOrigin(elem, inverseParent),
      pos, current, tr, localTarget;
  if (!inverseParent) { return; }
  if ($.isNumeric(limit)) {
    tr = getElementTranslation(elem);
    pos = addVector(matrixVectorProduct(totalParentTransform, tr), origin);
    current = {
      pageX: pos[0],
      pageY: pos[1]
    };
    target = limitMovement(current, target, limit);
  }
  localTarget = matrixVectorProduct(inverseParent,
      subtractVector([target.pageX, target.pageY], origin));
  if (localx || localy) {
    var sy = elemOldScale(elem);
    localTarget[0] += localx * sy;
    localTarget[1] -= localy * sy;
  }
  return cssNum(localTarget[0]) + ' ' + cssNum(localTarget[1]);
}

function homeContainer(elem) {
  var container = elem.offsetParent;
  if (!container) {
    return document;
  }
  return container;
}

// Compute the home position and the turtle location in local turtle
// coordinates; return the local offset from the home position as
// an array of len 2.
function computePositionAsLocalOffset(elem, home) {
  if (!home) {
    home = $(homeContainer(elem)).pagexy();
  }
  var totalParentTransform = totalTransform2x2(elem.parentElement),
      inverseParent = inverse2x2(totalParentTransform),
      origin = getTurtleOrigin(elem, inverseParent),
      ts = readTurtleTransform(elem, true),
      localHome = inverseParent && matrixVectorProduct(inverseParent,
          subtractVector([home.pageX, home.pageY], origin)),
      isy = 1 / elemOldScale(elem);
  if (!inverseParent) { return; }
  return [(ts.tx - localHome[0]) * isy, (localHome[1] - ts.ty) * isy];
}

function convertLocalXyToPageCoordinates(elem, localxy) {
  var totalParentTransform = totalTransform2x2(elem.parentElement),
      ts = readTurtleTransform(elem, true),
      center = $(homeContainer(elem)).pagexy(),
      sy = elemOldScale(elem),
      result = [],
      pageOffset, j;
  for (j = 0; j < localxy.length; j++) {
    pageOffset = matrixVectorProduct(
        totalParentTransform, [localxy[j][0] * sy, -localxy[j][1] * sy]);
    result.push({ pageX: center.pageX + pageOffset[0],
                  pageY: center.pageY + pageOffset[1] });
  }
  return result;
}

// Uses getBoundingClientRect to figure out current position in page
// coordinates.  Works by backing out local transformation (and inverting
// any parent rotations and distortions) so that the bounding rect is
// rectilinear; then reapplies translation (under any parent distortion)
// to get the final x and y, returned as {pageX:, pagey:}.
function getCenterInPageCoordinates(elem) {
  if ($.isWindow(elem)) {
    return getRoundedCenterLTWH(
        $(global).scrollLeft(), $(global).scrollTop(), ww(), wh());
  } else if (elem.nodeType === 9 || elem == document.body) {
    return getRoundedCenterLTWH(0, 0, dw(), dh());
  }
  var state = getTurtleData(elem);
  if (state && state.quickpagexy && state.down && state.style) {
    return state.quickpagexy;
  }
  var tr = getElementTranslation(elem),
      totalParentTransform = totalTransform2x2(elem.parentElement),
      simple = isone2x2(totalParentTransform),
      inverseParent = simple ? null : inverse2x2(totalParentTransform),
      origin = getTurtleOrigin(elem, inverseParent),
      pos = addVector(matrixVectorProduct(totalParentTransform, tr), origin),
      result = { pageX: pos[0], pageY: pos[1] };
  if (state && simple && state.down && state.style && elem.classList &&
      elem.classList.contains('turtle')) {
    state.quickpagexy = result;
  }
  return result;
}

// The quickpagexy variable is an optimization that assumes
// parent coordinates do not change.  This function will clear
// the cache, and is used when we have a container that is moving.
function clearChildQuickLocations(elem) {
  if (elem.tagName != 'CANVAS' && elem.tagName != 'IMG') {
    $(elem).find('.turtle').each(function(j, e) {
      var s = $.data(e, 'turtleData');
      if (s) {
        s.quickpagexy = null;
        s.quickhomeorigin = null;
      }
    });
  }
}

function polyToVectorsOffset(poly, offset) {
  if (!poly) { return null; }
  var result = [], j = 0;
  for (; j < poly.length; ++j) {
    result.push([poly[j].pageX + offset[0], poly[j].pageY + offset[1]]);
  }
  return result;
}

// Uses getBoundingClientRect to figure out the corners of the
// transformed parallelogram in page coordinates.
function getCornersInPageCoordinates(elem, untransformed) {
  if ($.isWindow(elem)) {
    return getStraightRectLTWH(
        $(global).scrollLeft(), $(global).scrollTop(), ww(), wh());
  } else if (elem.nodeType === 9) {
    return getStraightRectLTWH(0, 0, dw(), dh());
  }
  var currentTransform = readTransformMatrix(elem) || [1, 0, 0, 1],
      totalParentTransform = totalTransform2x2(elem.parentElement),
      totalTransform = matrixProduct(totalParentTransform, currentTransform),
      inverseParent = inverse2x2(totalParentTransform),
      out = {},
      origin = getTurtleOrigin(elem, inverseParent, out),
      gbcr = out.gbcr,
      hull = polyToVectorsOffset(getTurtleData(elem).hull, origin) || [
        [gbcr.left, gbcr.top],
        [gbcr.left, gbcr.bottom],
        [gbcr.right, gbcr.bottom],
        [gbcr.right, gbcr.top]
      ];
  if (untransformed) {
    // Used by the turtleHull css getter hook.
    return $.map(hull, function(pt) {
      return { pageX: pt[0] - origin[0], pageY: pt[1] - origin[1] };
    });
  }
  return $.map(hull, function(pt) {
    var tpt = translatedMVP(totalTransform, pt, origin);
    return { pageX: tpt[0], pageY: tpt[1] };
  });
}

function getDirectionOnPage(elem) {
  var ts = readTurtleTransform(elem, true),
      r = convertToRadians(normalizeRotation(ts.rot)),
      ux = Math.sin(r), uy = Math.cos(r),
      totalParentTransform = totalTransform2x2(elem.parentElement),
      up = matrixVectorProduct(totalParentTransform, [ux, uy]),
      dp = Math.atan2(up[0], up[1]);
  return radiansToDegrees(dp);
}

function scrollWindowToDocumentPosition(pos, limit) {
  var tx = pos.pageX,
      ty = pos.pageY,
      ww2 = ww() / 2,
      wh2 = wh() / 2,
      b = $('body'),
      dw = b.width(),
      dh = b.height(),
      w = $(global);
  if (tx > dw - ww2) { tx = dw - ww2; }
  if (tx < ww2) { tx = ww2; }
  if (ty > dh - wh2) { ty = dh - wh2; }
  if (ty < wh2) { ty = wh2; }
  var targ = { pageX: tx, pageY: ty };
  if ($.isNumeric(limit)) {
    targ = limitMovement(w.origin(), targ, limit);
  }
  w.scrollLeft(targ.pageX - ww2);
  w.scrollTop(targ.pageY - wh2);
}

//////////////////////////////////////////////////////////////////////////
// HIT DETECTION AND COLLISIONS
// Deal with touching and enclosing element rectangles taking
// into account distortions from transforms.
//////////////////////////////////////////////////////////////////////////

function signedTriangleArea(pt0, pt1, pt2) {
  var x1 = pt1.pageX - pt0.pageX,
      y1 = pt1.pageY - pt0.pageY,
      x2 = pt2.pageX - pt0.pageX,
      y2 = pt2.pageY - pt0.pageY;
  return x2 * y1 - x1 * y2;
}

function signedDeltaTriangleArea(pt0, diff1, pt2) {
  var x2 = pt2.pageX - pt0.pageX,
      y2 = pt2.pageY - pt0.pageY;
  return x2 * diff1.pageY - diff1.pageX * y2;
}

function pointInConvexPolygon(pt, poly) {
  // Implements top google hit algorithm for
  // ["An efficient test for a point to be in a convex polygon"]
  if (poly.length <= 0) { return false; }
  if (poly.length == 1) {
    return poly[0].pageX == pt.pageX && poly[0].pageY == pt.pageY;
  }
  var a0 = signedTriangleArea(pt, poly[poly.length - 1], poly[0]);
  if (a0 === 0) { return true; }
  var positive = (a0 > 0);
  if (poly.length == 2) { return false; }
  for (var j = 1; j < poly.length; ++j) {
    var aj = signedTriangleArea(pt, poly[j - 1], poly[j]);
    if (aj === 0) { return true; }
    if ((aj > 0) != positive) { return false; }
  }
  return true;
}

function diff(v1, v0) {
  return { pageX: v1.pageX - v0.pageX, pageY: v1.pageY - v0.pageY };
}

// Given an edge [p0, p1] of polygon P, and the expected sign of [p0, p1, p]
// for p inside P, then determine if all points in the other poly have the
// opposite sign.
function edgeSeparatesPointAndPoly(inside, p0, p1, poly) {
  var d1 = diff(p1, p0), j, s;
  for (j = 0; j < poly.length; ++j) {
    s = sign(signedDeltaTriangleArea(p0, d1, poly[j]));
    if (!s || s === inside) { return false; }
  }
  return true;
}

function sign(n) {
  return n > 0 ? 1 : n < 0 ? -1 : 0;
}

function convexPolygonSign(poly) {
  if (poly.length <= 2) { return 0; }
  var a = signedTriangleArea(poly[poly.length - 1], poly[0], poly[1]);
  if (a !== 0) { return sign(a); }
  for (var j = 1; j < poly.length; ++j) {
    a = signedTriangleArea(poly[j - 1], poly[j], poly[(j + 1) % poly.length]);
    if (a !== 0) { return sign(a); }
  }
  return 0;
}

function doConvexPolygonsOverlap(poly1, poly2) {
  // Implements top google hit for
  // ["polygon collision" gpwiki]
  var sign = convexPolygonSign(poly1), j;
  for (j = 0; j < poly1.length; ++j) {
    if (edgeSeparatesPointAndPoly(
        sign, poly1[j], poly1[(j + 1) % poly1.length], poly2)) {
      return false;
    }
  }
  sign = convexPolygonSign(poly2);
  for (j = 0; j < poly2.length; ++j) {
    if (edgeSeparatesPointAndPoly(
        sign, poly2[j], poly2[(j + 1) % poly2.length], poly1)) {
      return false;
    }
  }
  return true;
}

function doesConvexPolygonContain(polyOuter, polyInner) {
  // Just verify all vertices of polyInner are inside.
  for (var j = 0; j < polyInner.length; ++j) {
    if (!pointInConvexPolygon(polyInner[j], polyOuter)) {
      return false;
    }
  }
  return true;
}

// Google search for [Graham Scan Tom Switzer].
function convexHull(points) {
  function keepLeft(hull, r) {
    if (!r || !isPageCoordinate(r)) { return hull; }
    while (hull.length > 1 && sign(signedTriangleArea(hull[hull.length - 2],
        hull[hull.length - 1], r)) != 1) { hull.pop(); }
    if (!hull.length || !equalPoint(hull[hull.length - 1], r)) { hull.push(r); }
    return hull;
  }
  function reduce(arr, valueInitial, fnReduce) {
    for (var j = 0; j < arr.length; ++j) {
      valueInitial = fnReduce(valueInitial, arr[j]);
    }
    return valueInitial;
  }
  function equalPoint(p, q) {
    return p.pageX === q.pageX && p.pageY === q.pageY;
  }
  function lexicalPointOrder(p, q) {
    return p.pageX < q.pageX ? -1 : p.pageX > q.pageX ? 1 :
           p.pageY < q.pageY ? -1 : p.pageY > q.pageY ? 1 : 0;
  }
  points.sort(lexicalPointOrder);
  var leftdown = reduce(points, [], keepLeft),
      rightup = reduce(points.reverse(), [], keepLeft);
  return leftdown.concat(rightup.slice(1, -1));
}

function parseTurtleHull(text) {
  if (!text) return null;
  if ($.isArray(text)) return text;
  var nums = $.map(text.trim().split(/\s+/), parseFloat), points = [], j = 0;
  while (j + 1 < nums.length) {
    points.push({ pageX: nums[j], pageY: nums[j + 1] });
    j += 2;
  }
  return points;
}

function readTurtleHull(elem) {
  return getTurtleData(elem).hull;
}

function writeTurtleHull(hull) {
  for (var j = 0, result = []; j < hull.length; ++j) {
    result.push(hull[j].pageX, hull[j].pageY);
  }
  return result.length ? $.map(result, cssNum).join(' ') : 'none';
}

function makeHullHook() {
  // jQuery CSS hook for turtleHull property.
  return {
    get: function(elem, computed, extra) {
      var hull = getTurtleData(elem).hull;
      return writeTurtleHull(hull ||
          getCornersInPageCoordinates(elem, true));
    },
    set: function(elem, value) {
      var hull =
        !value || value == 'auto' ? null :
        value == 'none' ? [] :
        convexHull(parseTurtleHull(value));
      getTurtleData(elem).hull = hull;
    }
  };
}

//////////////////////////////////////////////////////////////////////////
// TURTLE CSS CONVENTIONS
// For better performance, the turtle library always writes transform
// CSS in a canonical form; and it reads this form faster than generic
// matrices.
//////////////////////////////////////////////////////////////////////////

// The canonical 2D transforms written by this plugin have the form:
// translate(tx, ty) rotate(rot) scale(sx, sy) rotate(twi)
// (with each component optional).
// This function quickly parses this form into a canonicalized object.
function parseTurtleTransform(transform) {
  if (transform === 'none') {
    return {tx: 0, ty: 0, rot: 0, sx: 1, sy: 1, twi: 0};
  }
  // Note that although the CSS spec doesn't allow 'e' in numbers, IE10
  // and FF put them in there; so allow them.
  var e = /^(?:translate\(([\-+.\de]+)(?:px)?,\s*([\-+.\de]+)(?:px)?\)\s*)?(?:rotate\(([\-+.\de]+)(?:deg)?\)\s*)?(?:scale\(([\-+.\de]+)(?:,\s*([\-+.\de]+))?\)\s*)?(?:rotate\(([\-+.\de]+)(?:deg)?\)\s*)?$/.exec(transform);
  if (!e) { return null; }
  var tx = e[1] ? parseFloat(e[1]) : 0,
      ty = e[2] ? parseFloat(e[2]) : 0,
      rot = e[3] ? parseFloat(e[3]) : 0,
      sx = e[4] ? parseFloat(e[4]) : 1,
      sy = e[5] ? parseFloat(e[5]) : sx,
      twi = e[6] ? parseFloat(e[6]) : 0;
  return {tx:tx, ty:ty, rot:rot, sx:sx, sy:sy, twi:twi};
}

function computeTurtleTransform(elem) {
  var m = readTransformMatrix(elem), d;
  if (!m) {
    return {tx: 0, ty: 0, rot: 0, sx: 1, sy: 1, twi: 0};
  }
  d = decomposeSVD(m);
  return {
    tx: m[4], ty: m[5], rot: radiansToDegrees(d[0]),
    sx: d[1], sy: d[2], twi: radiansToDegrees(d[3])
  };
}

function readTurtleTransform(elem, computed) {
  return parseTurtleTransform(elem.style[transform]) ||
      (computed && computeTurtleTransform(elem));
}

function cssNum(n) {
  var r = n.toString();
  if (~r.indexOf('e')) {
    r = Number(n).toFixed(17);
  }
  return r;
}

function writeTurtleTransform(ts) {
  var result = [];
  if (nonzero(ts.tx) || nonzero(ts.ty)) {
    result.push(
      'translate(' + cssNum(ts.tx) + 'px, ' + cssNum(ts.ty) + 'px)');
  }
  if (nonzero(ts.rot) || nonzero(ts.twi)) {
    result.push('rotate(' + cssNum(ts.rot) + 'deg)');
  }
  if (nonzero(1 - ts.sx) || nonzero(1 - ts.sy)) {
    if (nonzero(ts.sx - ts.sy)) {
      result.push('scale(' + cssNum(ts.sx) + ', ' + cssNum(ts.sy) + ')');
    } else {
      result.push('scale(' + cssNum(ts.sx) + ')');
    }
  }
  if (nonzero(ts.twi)) {
    result.push('rotate(' + cssNum(ts.twi) + 'deg)');
  }
  if (!result.length) {
    return 'none';
  }
  return result.join(' ');
}

function modulo(n, m) { return (+n % (m = +m) + m) % m; }

function radiansToDegrees(r) {
  var d = r * 180 / Math.PI;
  if (d > 180) { d -= 360; }
  return d;
}

function convertToRadians(d) {
  return d / 180 * Math.PI;
}

function normalizeRotation(x) {
  if (Math.abs(x) > 180) {
    x = x % 360;
    if (x > 180) { x -= 360; }
    else if (x <= -180) { x += 360; }
  }
  return x;
}

function normalizeRotationDelta(x) {
  if (Math.abs(x) >= 720) {
    x = x % 360 + (x > 0 ? 360 : -360);
  }
  return x;
}

//////////////////////////////////////////////////////////////////////////
// TURTLE DRAWING SUPPORT
// If pen, fill, or dot are used, then a full-page canvas is created
// and used for drawing.
//////////////////////////////////////////////////////////////////////////

// drawing state.
var globalDrawing = {
  attached: false,
  surface: null,
  field: null,
  ctx: null,
  canvas: null,
  timer: null,
  fieldMouse: false,  // if a child of the field listens to mouse events.
  fieldHook: false,   // once the body-event-forwarding logic is set up.
  subpixel: 1
};

function getTurtleField() {
  if (!globalDrawing.field) {
    createSurfaceAndField();
  }
  return globalDrawing.field;
}

function getTurtleClipSurface() {
  if (!globalDrawing.surface) {
    createSurfaceAndField();
  }
  return globalDrawing.surface;

}

function createSurfaceAndField() {
  var surface = document.createElement('samp'),
      field = document.createElement('samp'),
      cw = Math.floor(ww() / 2),
      ch = Math.floor(wh() / 2);
  $(surface)
    .css({
      position: 'absolute',
      display: 'inline-block',
      top: 0, left: 0, width: '100%', height: '100%',
      font: 'inherit',
      // z-index: -1 is required to keep the turtle
      // surface behind document text and buttons, so
      // the canvas does not block interaction
      zIndex: -1,
      // Setting transform origin for the turtle field
      // fixes a "center" point in page coordinates that
      // will not change even if the document resizes.
      transformOrigin: cw + "px " + ch + "px",
      pointerEvents: 'none',
      overflow: 'hidden'
    }).addClass('turtlefield');
  $(field).attr('id', 'origin')
    .css({
      position: 'absolute',
      display: 'inline-block',
      // Setting with to 100% allows label text to not wrap.
      top: ch, left: cw, width: '100%', height: '0',
      font: 'inherit',
      // Setting transform origin for the turtle field
      // fixes a "center" point in page coordinates that
      // will not change even if the document resizes.
      transformOrigin: "0px 0px",
      pointerEvents: 'all',
      // Setting turtleSpeed to Infinity by default allows
      // moving the origin instantly without sync.
      turtleSpeed: Infinity
    }).appendTo(surface);
  globalDrawing.surface = surface;
  globalDrawing.field = field;
  attachClipSurface();
  // Now that we have a surface, the upward-center cartesian coordinate
  // system based on that exists, so we can hook mouse events to add x, y.
  addMouseEventHooks();
}

function attachClipSurface() {
  if (document.body) {
    if ($('html').attr('style') == null) {
      // This prevents the body from shrinking.
      $('html').css('min-height', '100%');
    }
    $(globalDrawing.surface).prependTo('body');
    // Attach an event handler to forward mouse events from the body
    // to turtles in the turtle field layer.
    forwardBodyMouseEventsIfNeeded();
  } else {
    $(document).ready(attachClipSurface);
  }
}

// Given a $.data(elem, 'turtleData') state object, return or create
// the drawing canvas that this turtle should be drawing on.
function getDrawOnCanvas(state) {
  if (!state.drawOnCanvas) {
    state.drawOnCanvas = getTurtleDrawingCanvas();
  }
  return state.drawOnCanvas;
}

// Similar to getDrawOnCanvas, but for the read-only case: it avoids
// creating turtleData if none exists, and it avoid creating the global
// canvas if one doesn't already exist.  If there is no global canvas,
// this returns null.
function getCanvasForReading(elem) {
  var state = $.data(elem, 'turtleData');
  if (!state) return null;
  if (state.drawOnCanvas) return state.drawOnCanvas;
  return globalDrawing.canvas;
}

function getTurtleDrawingCanvas() {
  if (globalDrawing.canvas) {
    return globalDrawing.canvas;
  }
  var surface = getTurtleClipSurface();
  globalDrawing.canvas = document.createElement('canvas');
  $(globalDrawing.canvas).css({'z-index': -1});
  surface.insertBefore(globalDrawing.canvas, surface.firstChild);
  resizecanvas();
  pollbodysize(resizecanvas);
  $(global).resize(resizecanvas);
  return globalDrawing.canvas;
}

function getOffscreenCanvas(width, height) {
  if (globalDrawing.offscreen &&
      globalDrawing.offscreen.width === width &&
      globalDrawing.offscreen.height === height) {
    // Return a clean canvas.
    globalDrawing.offscreen.getContext('2d').clearRect(0, 0, width, height);
    return globalDrawing.offscreen;
  }
  if (!globalDrawing.offscreen) {
    globalDrawing.offscreen = document.createElement('canvas');
    /* for debugging "touches": make offscreen canvas visisble.
    $(globalDrawing.offscreen)
      .css({position:'absolute',top:0,left:0,zIndex:1})
      .appendTo('body');
    */
  }
  globalDrawing.offscreen.width = width;
  globalDrawing.offscreen.height = height;
  return globalDrawing.offscreen;
}

function pollbodysize(callback) {
  var b = $('body');
  var lastwidth = b.width();
  var lastheight = b.height();
  var poller = (function() {
    if (b.width() != lastwidth || b.height() != lastheight) {
      callback();
      lastwidth = b.width();
      lastheight = b.height();
    }
  });
  if (globalDrawing.timer) {
    clearInterval(globalDrawing.timer);
  }
  globalDrawing.timer = setInterval(poller, 250);
}

function sizexy() {
  // Notice that before the body exists, we cannot get its size; so
  // we fall back to the window size.
  // Using innerHeight || $(window).height() deals with quirks-mode.
  var b = $('body');
  return [
    Math.max(b.outerWidth(true), global.innerWidth || $(global).width()),
    Math.max(b.outerHeight(true), global.innerHeight || $(global).height())
  ];
}

function resizecanvas() {
  if (!globalDrawing.canvas) return;
  var sxy = sizexy(),
      ww = sxy[0],
      wh = sxy[1],
      cw = globalDrawing.canvas.width,
      ch = globalDrawing.canvas.height,
      // Logic: minimum size 200; only shrink if larger than 2000;
      // and only resize if changed more than 100 pixels.
      bw = Math.max(Math.min(2000, Math.max(200, cw)),
                    Math.ceil(ww / 100) * 100) * globalDrawing.subpixel,
      bh = Math.max(Math.min(2000, Math.max(200, cw)),
                    Math.ceil(wh / 100) * 100) * globalDrawing.subpixel,
      tc;
  $(globalDrawing.surface).css({
      width: ww + 'px',
      height: wh + 'px'
  });
  if (cw != bw || ch != bh) {
    // Transfer canvas out to tc and back again after resize.
    tc = document.createElement('canvas');
    tc.width = Math.min(cw, bw);
    tc.height = Math.min(ch, bh);
    tc.getContext('2d').drawImage(globalDrawing.canvas, 0, 0);
    globalDrawing.canvas.width = bw;
    globalDrawing.canvas.height = bh;
    globalDrawing.canvas.getContext('2d').drawImage(tc, 0, 0);
    $(globalDrawing.canvas).css({
      width: bw / globalDrawing.subpixel,
      height: bh / globalDrawing.subpixel
    });
  }
}

// turtlePenStyle style syntax
function parsePenStyle(text, defaultProp) {
  if (!text) { return null; }
  if (text && (typeof(text) == "function") && (
      text.helpname || text.name)) {
    // Deal with "tan" and "fill".
    text = (text.helpname || text.name);
  }
  text = String(text);
  if (text.trim) { text = text.trim(); }
  if (!text || text === 'none') { return null; }
  if (text === 'path' || text === 'fill') {
    return { savePath: true };
  }
  var eraseMode = false;
  if (/^erase\b/.test(text)) {
    text = text.replace(
        /^erase\b/, 'white; globalCompositeOperation:destination-out');
    eraseMode = true;
  }
  var result = parseOptionString(text, defaultProp);
  if (eraseMode) { result.eraseMode = true; }
  return result;
}

function writePenStyle(style) {
  if (!style) { return 'none'; }
  return printOptionAsString(style);
}

function parsePenDown(style) {
  if (style == 'down' || style === true) return true;
  if (style == 'up' || style === false) return false;
  return undefined;
}

function writePenDown(bool) {
  return bool ? 'down' : 'up';
}

function getTurtleData(elem) {
  var state = $.data(elem, 'turtleData');
  if (!state) {
    state = $.data(elem, 'turtleData', {
      style: null,
      corners: [[]],
      path: [[]],
      down: false,
      speed: 'turtle',
      easing: 'swing',
      turningRadius: 0,
      drawOnCanvas: null,
      quickpagexy: null,
      quickhomeorigin: null,
      oldscale: 1,
      instrument: null,
      stream: null
    });
  }
  return state;
}

function getTurningRadius(elem) {
  var state = $.data(elem, 'turtleData');
  if (!state) { return 0; }
  return state.turningRadius;
}

function makeTurningRadiusHook() {
  return {
    get: function(elem, computed, extra) {
      return cssNum(getTurningRadius(elem)) + 'px';
    },
    set: function(elem, value) {
      var radius = parseFloat(value);
      if (isNaN(radius)) return;
      getTurtleData(elem).turningRadius = radius;
      elem.style.turtleTurningRadius = '' + cssNum(radius) + 'px';
      if (radius === 0) {
        // When radius goes to zero, renormalize rotation to
        // between 180 and -180.  (We avoid normalizing rotation
        // when there is a visible turning radius so we can tell
        // the difference between +361 and +1 and -359 arcs,
        // which are all different.)
        var ts = readTurtleTransform(elem, false);
        if (ts && (ts.rot > 180 || ts.rot <= -180)) {
          ts.rot = normalizeRotation(ts.rot);
          elem.style[transform] = writeTurtleTransform(ts);
        }
      }
    }
  };
}

function makePenStyleHook() {
  return {
    get: function(elem, computed, extra) {
      return writePenStyle(getTurtleData(elem).style);
    },
    set: function(elem, value) {
      var style = parsePenStyle(value, 'strokeStyle'),
          state = getTurtleData(elem);
      if (state.style) {
        // Switch to an empty pen first, to terminate paths.
        state.style = null;
        flushPenState(elem, state, true);
      }
      state.style = style;
      elem.style.turtlePenStyle = writePenStyle(style);
      flushPenState(elem, state, true);
    }
  };
}

function makePenDownHook() {
  return {
    get: function(elem, computed, extra) {
      return writePenDown(getTurtleData(elem).down);
    },
    set: function(elem, value) {
      var style = parsePenDown(value);
      if (style === undefined) return;
      var state = getTurtleData(elem);
      if (style != state.down) {
        state.down = style;
        state.quickpagexy = null;
        state.quickhomeorigin = null;
        elem.style.turtlePenDown = writePenDown(style);
        flushPenState(elem, state, true);
      }
    }
  };
}

function isPointNearby(a, b) {
  return Math.round(a.pageX - b.pageX) === 0 &&
         Math.round(a.pageY - b.pageY) === 0;
}

function isPointVeryNearby(a, b) {
  return Math.round(1000 * (a.pageX - b.pageX)) === 0 &&
         Math.round(1000 * (a.pageY - b.pageY)) === 0;
}

function isBezierTiny(a, b) {
  return isPointNearby(a, b) &&
         Math.round(a.pageX - b.pageX1) === 0 &&
         Math.round(a.pageY - b.pageY1) === 0 &&
         Math.round(b.pageX2 - b.pageX) === 0 &&
         Math.round(b.pageY2 - b.pageY) === 0;
}

function roundEpsilon(x) {
  var dig3 = x * 1000, tru3 = Math.round(dig3);
  if (Math.abs(tru3 - dig3) < Math.abs(5e-15 * dig3)) {
    return tru3 / 1000;
  }
  return x;
}

function applyPenStyle(ctx, ps, scale) {
  scale = scale || 1;
  var extraWidth = ps.eraseMode ? 1 : 0;
  if (!ps || !('strokeStyle' in ps)) { ctx.strokeStyle = 'black'; }
  if (!ps || !('lineWidth' in ps)) {
    ctx.lineWidth = 1.62 * scale + extraWidth;
  }
  if (!ps || !('lineCap' in ps)) { ctx.lineCap = 'round'; }
  if (!ps || !('lineJoin' in ps)) { ctx.lineJoin = 'round'; }
  if (ps) {
    for (var a in ps) {
      if (a === 'savePath' || a === 'eraseMode') { continue; }
      if (scale && a === 'lineWidth') {
        ctx[a] = scale * ps[a] + extraWidth;
      } else if (a === 'lineDash') {
        ctx.setLineDash(('' + ps[a]).split(/[,\s]/g));
      } else {
        ctx[a] = ps[a];
      }
    }
  }
}

// Computes a matrix that transforms page coordinates to the local
// canvas coordinates.  Applying this matrix as the canvas transform
// allows us to draw on a canvas using page coordinates; and the bits
// will show up on the canvas in the corresponding location on the
// physical page, even if the canvas has been moved by absolute
// position and CSS 2d transforms.
function computeCanvasPageTransform(canvas) {
  if (!canvas) { return; }
  if (canvas === globalDrawing.canvas) {
    return [globalDrawing.subpixel, 0, 0, globalDrawing.subpixel];
  }
  var totalParentTransform = totalTransform2x2(canvas.parentElement),
      inverseParent = inverse2x2(totalParentTransform),
      out = {},
      origin = getTurtleOrigin(canvas, inverseParent, out),
      gbcr = out.gbcr,
      originTranslate = [1, 0, 0, 1, -origin[0], -origin[1]],
      finalScale = gbcr.width && gbcr.height &&
          [canvas.width / gbcr.width, 0, 0, canvas.height / gbcr.height],
      localTransform = readTransformMatrix(canvas) || [1, 0, 0, 1],
      inverseTransform = inverse2x3(localTransform),
      totalInverse;
  if (!inverseParent || !inverseTransform || !finalScale) {
    return;
  }
  totalInverse =
      matrixProduct(
        matrixProduct(
          matrixProduct(
            finalScale,
            inverseTransform),
          inverseParent),
        originTranslate);
  totalInverse[4] += out.localorigin[0] * finalScale[0];
  totalInverse[5] += out.localorigin[1] * finalScale[3];
  return totalInverse;
}

function setCanvasPageTransform(ctx, canvas) {
  if (canvas === globalDrawing.canvas) {
    ctx.setTransform(
        globalDrawing.subpixel, 0, 0, globalDrawing.subpixel, 0, 0);
  } else {
    var pageToCanvas = computeCanvasPageTransform(canvas);
    if (pageToCanvas) {
      ctx.setTransform.apply(ctx, pageToCanvas);
    }
  }
}

var buttOverlap = 0.67;

function drawAndClearPath(drawOnCanvas, path, style, scale, truncateTo) {
  var ctx = drawOnCanvas.getContext('2d'),
      isClosed, skipLast,
      j = path.length,
      segment;
  ctx.save();
  setCanvasPageTransform(ctx, drawOnCanvas);
  ctx.beginPath();
  // Scale up lineWidth by sx.  (TODO: consider parent transforms.)
  applyPenStyle(ctx, style, scale);
  while (j--) {
    if (path[j].length > 1) {
      segment = path[j];
      isClosed = segment.length > 2 &&
          isPointNearby(segment[0], segment[segment.length - 1]) &&
          !isPointNearby(segment[0], segment[Math.floor(segment.length / 2)]);
      skipLast = isClosed && (!('pageX2' in segment[segment.length - 1]));
      var startx = segment[0].pageX;
      var starty = segment[0].pageY;
      ctx.moveTo(startx, starty);
      for (var k = 1; k < segment.length - (skipLast ? 1 : 0); ++k) {
        if ('pageX2' in segment[k] &&
            !isBezierTiny(segment[k - 1], segment[k])) {
          ctx.bezierCurveTo(
             segment[k].pageX1, segment[k].pageY1,
             segment[k].pageX2, segment[k].pageY2,
             segment[k].pageX, segment[k].pageY);
        } else {
          ctx.lineTo(segment[k].pageX, segment[k].pageY);
        }
      }
      if (isClosed) { ctx.closePath(); }
    }
  }
  if ('fillStyle' in style) { ctx.fill(); }
  if ('strokeStyle' in style) { ctx.stroke(); }
  ctx.restore();
  path.length = 1;
  path[0].splice(0, Math.max(0, path[0].length - truncateTo));
}

function addBezierToPath(path, start, triples) {
  if (!path.length || !isPointNearby(start, path[path.length - 1])) {
    path.push(start);
  }
  for (var j = 0; j < triples.length; ++j) {
    path.push({
        pageX1: triples[j][0].pageX, pageY1: triples[j][0].pageY,
        pageX2: triples[j][1].pageX, pageY2: triples[j][1].pageY,
        pageX: triples[j][2].pageX, pageY: triples[j][2].pageY });
  }
}

function addToPathList(pathList, point) {
  if (pathList.length &&
      (point.corner ? isPointVeryNearby(point, pathList[pathList.length - 1]) :
                      isPointNearby(point, pathList[pathList.length - 1]))) {
    return;
  }
  pathList.push(point);
}

function flushPenState(elem, state, corner) {
  clearChildQuickLocations(elem);
  if (!state) {
    // Default is no pen and no path, so nothing to do.
    return;
  }
  var path = state.path,
      style = state.style,
      corners = state.corners;
  if (!style || !state.down) {
    // pen up or pen null will clear the tracing path.
    if (path.length > 1) { path.length = 1; }
    if (path[0].length) { path[0].length = 0; }
    if (corner) {
      if (!style) {
        // pen null will clear the retracing path too.
        if (corners.length > 1) corners.length = 1;
        if (corners[0].length) corners[0].length = 0;
      } else {
        // pen up with a non-null pen will start a new discontinuous segment.
        if (corners.length && corners[0].length) {
          if (corners[0].length == 1) {
            corners[0].length = 0;
          } else {
            corners.unshift([]);
          }
        }
      }
    }
    return;
  }
  if (!corner && style.savePath) return;
  // Accumulate retracing path using only corners.
  var center = getCenterInPageCoordinates(elem);
  if (corner) {
    center.corner = true;
    addToPathList(corners[0], center);
  }
  if (style.savePath) return;
  // Add to tracing path, and trace it right away.
  addToPathList(path[0], center);
  var scale = drawingScale(elem);
  // Last argument 2 means that the last two points are saved, which
  // allows us to draw corner miters and also avoid 'butt' lineCap gaps.
  drawAndClearPath(getDrawOnCanvas(state), state.path, style, scale, 2);
}

function endAndFillPenPath(elem, style) {
  var state = getTurtleData(elem);
  if (state.style) {
    // Apply a default style.
    style = $.extend({}, state.style, style);
  }
  var scale = drawingScale(elem);
  drawAndClearPath(getDrawOnCanvas(state), state.corners, style, scale, 1);
}

function clearField(arg) {
  if ((!arg || /\bcanvas\b/.test(arg)) && globalDrawing.canvas) {
    var ctx = globalDrawing.canvas.getContext('2d');
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(
        0, 0, globalDrawing.canvas.width, globalDrawing.canvas.height);
    ctx.restore();
  }
  if (!arg || /\bturtles\b/.test(arg)) {
    if (globalDrawing.surface) {
      var sel = $(globalDrawing.surface).find('.turtle').not('.turtlefield');
      if (global_turtle) {
        sel = sel.not(global_turtle);
      }
      sel.remove();
    }
  }
  if (!arg || /\blabels\b/.test(arg)) {
    if (globalDrawing.surface) {
      var sel = $(globalDrawing.surface).find('.turtlelabel')
                .not('.turtlefield');
      sel.remove();
    }
  }
  if (!arg || /\btext\b/.test(arg)) {
    // "turtlefield" is a CSS class to use to mark elements that
    // should not be deleted by clearscreen.
    $('body').contents().not('.turtlefield').remove();
  }
}

function getBoundingBoxOfCorners(c, clip) {
  if (!c || c.length < 1) return null;
  var j = 1, result = {
    left: Math.floor(c[0].pageX),
    top: Math.floor(c[0].pageY),
    right: Math.ceil(c[0].pageX),
    bottom: Math.ceil(c[0].pageY)
  };
  for (; j < c.length; ++j) {
    result.left = Math.min(result.left, Math.floor(c[j].pageX));
    result.top = Math.min(result.top, Math.floor(c[j].pageY));
    result.right = Math.max(result.right, Math.ceil(c[j].pageX));
    result.bottom = Math.max(result.bottom, Math.ceil(c[j].pageY));
  }
  if (clip) {
    result.left = Math.max(clip.left, result.left);
    result.top = Math.max(clip.top, result.top);
    result.right = Math.min(clip.right, result.right);
    result.bottom = Math.min(clip.bottom, result.bottom);
  }
  return result;
}

function transformPoints(m, points) {
  if (!m || !points) return null;
  if (m.length == 4 && isone2x2(m)) return points;
  var result = [], j, prod;
  for (j = 0; j < points.length; ++j) {
    prod = matrixVectorProduct(m, [points[j].pageX, points[j].pageY]);
    result.push({pageX: prod[0], pageY: prod[1]});
  }
  return result;
}

function touchesPixel(elem, color) {
  if (!elem) { return false; }
  var rgba = rgbaForColor(color),
      canvas = getCanvasForReading(elem);
  if (!canvas) { return rgba && rgba[3] == 0; }
  var trans = computeCanvasPageTransform(canvas),
      originalc = getCornersInPageCoordinates(elem),
      c = transformPoints(trans, originalc),
      bb = getBoundingBoxOfCorners(c,
          {left:0, top:0, right:canvas.width, bottom:canvas.height}),
      w = (bb.right - bb.left),
      h = (bb.bottom - bb.top),
      osc = getOffscreenCanvas(w, h),
      octx = osc.getContext('2d'),
      j = 1, k;
  if (!c || c.length < 3 || !w || !h) { return false; }
  octx.drawImage(canvas,
      bb.left, bb.top, w, h, 0, 0, w, h);
  octx.save();
  // Erase everything outside clipping region.
  octx.beginPath();
  octx.moveTo(0, 0);
  octx.lineTo(w, 0);
  octx.lineTo(w, h);
  octx.lineTo(0, h);
  octx.closePath();
  octx.moveTo((c[0].pageX - bb.left),
              (c[0].pageY - bb.top));
  for (; j < c.length; j += 1) {
    octx.lineTo((c[j].pageX - bb.left),
                (c[j].pageY - bb.top));
  }
  octx.closePath();
  octx.clip();
  if (rgba && rgba[3] == 0) {
    // If testing for transparent, should clip with black, not transparent.
    octx.fillRect(0, 0, w, h);
  } else {
    octx.clearRect(0, 0, w, h);
  }
  octx.restore();
  // Now examine the results and look for alpha > 0%.
  var data = octx.getImageData(0, 0, w, h).data;
  if (!rgba || rgba[3] == 0) {
    // Handle the "looking for any color" and "transparent" cases.
    var wantcolor = !rgba;
    for (j = 0; j < data.length; j += 4) {
      if ((data[j + 3] > 0) == wantcolor) return true;
    }
  } else {
    for (j = 0; j < data.length; j += 4) {
      // Look for a near-match in color: within a 7x7x7 cube in rgb space,
      // and at least 50% of the target alpha value.
      if (Math.abs(data[j + 0] - rgba[0]) <= 3 &&
          Math.abs(data[j + 1] - rgba[1]) <= 3 &&
          Math.abs(data[j + 2] - rgba[2]) <= 3 &&
          data[j + 3] <= rgba[3] * 2 && data[j + 3] >= rgba[3] / 2) {
        return true;
      }
    }
  }
  return false;
}

//////////////////////////////////////////////////////////////////////////
// JQUERY METHOD SUPPORT
// Functions in direct support of exported methods.
//////////////////////////////////////////////////////////////////////////

function applyImg(sel, img, cb) {
  if (img.img) {
    if (sel[0].tagName == 'CANVAS' || sel[0].tagName == img.img.tagName) {
      applyLoadedImage(img.img, sel[0], img.css);
    }
  } else if (sel[0].tagName == 'IMG' || sel[0].tagName == 'CANVAS') {
    setImageWithStableOrigin(sel[0], img.url, img.css, cb);
    cb = null;
  } else {
    var props = {
      backgroundImage: 'url(' + img.url + ')',
      backgroundRepeat: 'no-repeat',
      backgroundPosition: 'center',
    };
    if (img.css.width && img.css.height) {
      props.backgroundSize = img.css.width + 'px ' + img.css.height + 'px';
    }
    sel.css(props);
  }
  if (cb) {
    cb();
  }
}

function doQuickMove(elem, distance, sideways) {
  var ts = readTurtleTransform(elem, true),
      r = ts && convertToRadians(ts.rot),
      sy = elemOldScale(elem),
      scaledDistance = ts && (distance * sy),
      scaledSideways = ts && ((sideways || 0) * sy),
      dy = -Math.cos(r) * scaledDistance,
      dx = Math.sin(r) * scaledDistance,
      state = $.data(elem, 'turtleData'),
      qpxy;
  if (!ts) { return; }
  if (sideways) {
    dy += Math.sin(r) * scaledSideways;
    dx += Math.cos(r) * scaledSideways;
  }
  if (state && (qpxy = state.quickpagexy)) {
    state.quickpagexy = {
      pageX: qpxy.pageX + dx,
      pageY: qpxy.pageY + dy
    };
  }
  ts.tx += dx;
  ts.ty += dy;
  elem.style[transform] = writeTurtleTransform(ts);
  flushPenState(elem, state, true);
}

function doQuickMoveXY(elem, dx, dy) {
  var ts = readTurtleTransform(elem, true),
      state = $.data(elem, 'turtleData'),
      qpxy;
  if (!ts) { return; }
  if (state && (qpxy = state.quickpagexy)) {
    state.quickpagexy = {
      pageX: qpxy.pageX + dx,
      pageY: qpxy.pageY - dy
    };
  }
  ts.tx += dx;
  ts.ty -= dy;
  elem.style[transform] = writeTurtleTransform(ts);
  flushPenState(elem, state, true);
}

function doQuickRotate(elem, degrees) {
  var ts = readTurtleTransform(elem, true);
  if (!ts) { return; }
  ts.rot += degrees;
  elem.style[transform] = writeTurtleTransform(ts);
}

function displacedPosition(elem, distance, sideways) {
  var ts = readTurtleTransform(elem, true);
  if (!ts) { return; }
  var s = elemOldScale(elem),
      r = convertToRadians(ts.rot),
      scaledDistance = distance * s,
      scaledSideways = (sideways || 0) * s,
      dy = -Math.cos(r) * scaledDistance,
      dx = Math.sin(r) * scaledDistance;
  if (scaledSideways) {
    dy += Math.sin(r) * scaledSideways;
    dx += Math.cos(r) * scaledSideways;
  }
  return cssNum(ts.tx + dx) + ' ' + cssNum(ts.ty + dy);
}

function isPageCoordinate(obj) {
  return obj && $.isNumeric(obj.pageX) && $.isNumeric(obj.pageY);
}

function makeTurtleSpeedHook() {
  return {
    get: function(elem, computed, extra) {
      return getTurtleData(elem).speed;
    },
    set: function(elem, value) {
      if ((!$.isNumeric(value) || value <= 0) &&
          !(value in $.fx.speeds) && ('' + value != 'Infinity')) {
        return;
      }
      getTurtleData(elem).speed = '' + value;
    }
  }
}

function makeTurtleEasingHook() {
  return {
    get: function(elem, computed, extra) {
      return getTurtleData(elem).easing;
    },
    set: function(elem, value) {
      if (!(value in $.easing)) {
        return;
      }
      getTurtleData(elem).easing = value;
    }
  }
}

function animTime(elem, intick) {
  var state = $.data(elem, 'turtleData');
  intick = intick || insidetick;
  if (!state) return (intick ? 0 : 'turtle');
  if ($.isNumeric(state.speed) || state.speed == 'Infinity') {
    return 1000 / state.speed;
  }
  if (state.speed == 'turtle' && intick) return 0;
  return state.speed;
}

function animEasing(elem) {
  var state = $.data(elem, 'turtleData');
  if (!state) return null;
  return state.easing;
}

function makeTurtleForwardHook() {
  return {
    get: function(elem, computed, extra) {
      // TODO: after reading turtleForward, we need to also
      // adjust it if ts.tx/ty change due to an origin change,
      // so that images don't stutter if they resize during an fd.
      // OR - offset by origin, so that changes in its value are
      // not a factor.
      var ts = readTurtleTransform(elem, computed),
          middle = readTransformOrigin(elem);
      if (ts) {
        var r = convertToRadians(ts.rot),
            c = Math.cos(r),
            s = Math.sin(r),
            sy = elemOldScale(elem);
        return cssNum(((ts.tx + middle[0]) * s - (ts.ty + middle[1]) * c)
            / sy) + 'px';
      }
    },
    set: function(elem, value) {
      var ts = readTurtleTransform(elem, true) ||
              {tx: 0, ty: 0, rot: 0, sx: 1, sy: 1, twi: 0},
          middle = readTransformOrigin(elem),
          sy = elemOldScale(elem),
          v = parseFloat(value) * sy,
          r = convertToRadians(ts.rot),
          c = Math.cos(r),
          s = Math.sin(r),
          p = (ts.tx + middle[0]) * c + (ts.ty + middle[1]) * s,
          ntx = p * c + v * s - middle[0],
          nty = p * s - v * c - middle[1],
          state = $.data(elem, 'turtleData'),
          qpxy;
      if (state && (qpxy = state.quickpagexy)) {
        state.quickpagexy = {
          pageX: qpxy.pageX + (ntx - ts.tx),
          pageY: qpxy.pageY + (nty - ts.ty)
        };
      }
      ts.tx = ntx;
      ts.ty = nty;
      elem.style[transform] = writeTurtleTransform(ts);
      flushPenState(elem, state);
    }
  };
}

// Finally, add turtle support.
function makeTurtleHook(prop, normalize, unit, displace) {
  return {
    get: function(elem, computed, extra) {
      var ts = readTurtleTransform(elem, computed);
      if (ts) { return ts[prop] + unit; }
    },
    set: function(elem, value) {
      var ts = readTurtleTransform(elem, true) ||
          {tx: 0, ty: 0, rot: 0, sx: 1, sy: 1, twi: 0},
          opt = { displace: displace },
          state = $.data(elem, 'turtleData'),
          otx = ts.tx, oty = ts.ty, qpxy;
      ts[prop] = normalize(value, elem, ts, opt);
      elem.style[transform] = writeTurtleTransform(ts);
      if (opt.displace) {
        if (state && (qpxy = state.quickpagexy)) {
          state.quickpagexy = {
            pageX: qpxy.pageX + (ts.tx - otx),
            pageY: qpxy.pageY + (ts.ty - oty)
          };
        }
        flushPenState(elem, state);
      } else {
        clearChildQuickLocations(elem);
      }
    }
  };
}

// Given a starting direction, angle change, and turning radius,
// this computes the side-radius (with a sign flip indicating
// the other side), the coordinates of the center dc, and the dx/dy
// displacement of the final location after the arc.
function setupArc(
    r0,         // starting direction in local coordinates
    r1,         // ending direction local coordinates
    turnradius  // turning radius in local coordinates
) {
  var delta = normalizeRotationDelta(r1 - r0),
      sradius = delta > 0 ? turnradius : -turnradius,
      r0r = convertToRadians(r0),
      dc = [Math.cos(r0r) * sradius, Math.sin(r0r) * sradius],
      r1r = convertToRadians(r1);
  return {
    delta: delta,
    sradius: sradius,
    dc: dc,
    dx: dc[0] - Math.cos(r1r) * sradius,
    dy: dc[1] - Math.sin(r1r) * sradius
  };
}

// Given a path array, a pageX/pageY starting position,
// arc information in local coordinates, and a 2d transform
// between page and local coordinates, this function adds to
// the path scaled page-coorindate beziers following the arc.
function addArcBezierPaths(
    path,       // path to add on to in page coordinates
    start,      // starting location in page coordinates
    r0,         // starting direction in local coordinates
    end,        // ending direction in local coordinates
    turnradius, // turning radius in local coordinates
    transform   // linear distortion between page and local
) {
  var a = setupArc(r0, end, turnradius),
      sradius = a.sradius, dc = a.dc,
      r1, a1r, a2r, j, r, pts, triples,
      splits, splita, absang, relative, points;
  // Decompose an arc into equal arcs, all 45 degrees or less.
  splits = 1;
  splita = a.delta;
  absang = Math.abs(a.delta);
  if (absang > 45) {
    splits = Math.ceil(absang / 45);
    splita = a.delta / splits;
  }
  // Relative traces out the unit-radius arc centered at the origin.
  relative = [];
  while (--splits >= 0) {
    r1 = splits === 0 ? end : r0 + splita;
    a1r = convertToRadians(r0 + 180);
    a2r = convertToRadians(r1 + 180);
    relative.push.apply(relative, approxBezierUnitArc(a1r, a2r));
    r0 = r1;
  }
  points = [];
  for (j = 0; j < relative.length; j++) {
    // Multiply each coordinate by radius scale up to the right
    // turning radius and add to dc to center the turning radius
    // at the right local coordinate position; then apply parent
    // distortions to get page-coordinate relative offsets to the
    // turtle's original position.
    r = matrixVectorProduct(transform,
        addVector(scaleVector(relative[j], sradius), dc));
    // Finally add these to the turtle's actual original position
    // to get page-coordinate control points for the bezier curves.
    // (start is the starting position in absolute coordinates,
    // and dc is the local coordinate offset from the starting
    // position to the center of the turning radius.)
    points.push({
      pageX: r[0] + start.pageX,
      pageY: r[1] + start.pageY});
  }
  // Divide control points into triples again to form bezier curves.
  triples = [];
  for (j = 0; j < points.length; j += 3) {
    triples.push(points.slice(j, j + 3));
  }
  addBezierToPath(path, start, triples);
  return a;
}

// An animation driver for rotation, including the possibility of
// tracing out an arc.  Reads an element's turningRadius to see if
// changing ts.rot should also sweep out an arc.  If so, calls
// addArcBezierPath to directly add that arc to the drawing path.
function maybeArcRotation(end, elem, ts, opt) {
  end = parseFloat(end);
  var state = $.data(elem, 'turtleData'),
      tradius = state ? state.turningRadius : 0;
  if (tradius === 0 || ts.rot == end) {
    // Avoid drawing a line if zero turning radius.
    opt.displace = false;
    return tradius === 0 ? normalizeRotation(end) : end;
  }
  var tracing = (state && state.style && state.down),
      sy = (state && state.oldscale) ? ts.sy : 1,
      turnradius = tradius * sy, a;
  if (tracing) {
    a = addArcBezierPaths(
      state.path[0],                            // path to add to
      getCenterInPageCoordinates(elem),         // starting location
      ts.rot,                                   // starting direction
      end,                                      // ending direction
      turnradius,                               // scaled turning radius
      totalTransform2x2(elem.parentElement));   // totalParentTransform
  } else {
    a = setupArc(
      ts.rot,                                   // starting direction
      end,                                      // degrees change
      turnradius);                              // scaled turning radius
  }
  ts.tx += a.dx;
  ts.ty += a.dy;
  opt.displace = true;
  return end;
}

function makeRotationStep(prop) {
  return function(fx) {
    if (!fx.delta) {
      fx.delta = normalizeRotationDelta(fx.end - fx.start);
      fx.start = fx.end - fx.delta;
    }
    $.cssHooks[prop].set(fx.elem, fx.start + fx.delta * fx.pos);
  };
}

function splitPair(text, duplicate) {
  if (text.length && text[0] === '_') {
    // Hack: remove forced number non-conversion.
    text = text.substring(1);
  }
  var result = $.map(('' + text).split(/\s+/), parseFloat);
  while (result.length < 2) {
    result.push(duplicate ?
        (!result.length ? 1 : result[result.length - 1]) : 0);
  }
  return result;
}

function makePairStep(prop, displace) {
  return function(fx) {
    if (!fx.delta) {
      var end = splitPair(fx.end, !displace);
      fx.start = splitPair(fx.start, !displace);
      fx.delta = [end[0] - fx.start[0], end[1] - fx.start[1]];
    }
    $.cssHooks[prop].set(fx.elem, [fx.start[0] + fx.delta[0] * fx.pos,
        fx.start[1] + fx.delta[1] * fx.pos].join(' '));
  };
}

var XY = ['X', 'Y'];
function makeTurtleXYHook(publicname, propx, propy, displace) {
  return {
    get: function(elem, computed, extra) {
      var ts = readTurtleTransform(elem, computed);
      if (ts) {
        if (displace || ts[propx] != ts[propy]) {
          // Hack: if asked to convert a pair to a number by fx, then refuse.
          return (extra === '' ? '_' : '') + ts[propx] + ' ' + ts[propy];
        } else {
          return '' + ts[propx];
        }
      }
    },
    set: function(elem, value, extra) {
      var ts = readTurtleTransform(elem, true) ||
              {tx: 0, ty: 0, rot: 0, sx: 1, sy: 1, twi: 0},
          parts = (typeof(value) == 'string' ? value.split(/\s+/) : [value]),
          state = $.data(elem, 'turtleData'),
          otx = ts.tx, oty = ts.ty, qpxy;
      if (parts.length < 1 || parts.length > 2) { return; }
      if (parts.length >= 1) { ts[propx] = parseFloat(parts[0]); }
      if (parts.length >= 2) { ts[propy] = parseFloat(parts[1]); }
      else if (!displace) { ts[propy] = ts[propx]; }
      else { ts[propy] = 0; }
      elem.style[transform] = writeTurtleTransform(ts);
      if (displace) {
        if (state && (qpxy = state.quickpagexy)) {
          state.quickpagexy = {
            pageX: qpxy.pageX + (ts.tx - otx),
            pageY: qpxy.pageY + (ts.ty - oty)
          };
        }
        flushPenState(elem, state);
      } else {
        clearChildQuickLocations(elem);
      }
    }
  };
}

var absoluteUrlAnchor = document.createElement('a');
function absoluteUrlObject(url) {
  absoluteUrlAnchor.href = url;
  return absoluteUrlAnchor;
}
function absoluteUrl(url) {
  return absoluteUrlObject(url).href;
}

// Pencil-code specific function: detects whether a domain appears to
// be a pencilcode site.
function isPencilHost(hostname) {
  return /(?:^|\.)pencil(?:code)?\./i.test(hostname);
}
// Returns a pencilcode username from the URL, if any.
function pencilUserFromUrl(url) {
  var hostname = absoluteUrlObject(url == null ? '' : url).hostname,
      match = /^(\w+)\.pencil(?:code)?\./i.exec(hostname);
  if (match) return match[1];
  return null;
}
// Rewrites a url to have the top directory name given.
function apiUrl(url, topdir) {
  var link = absoluteUrlObject(url == null ? '' : url),
      result = link.href;
  if (isPencilHost(link.hostname)) {
    if (/^\/(?:edit|home|code|load|save)(?:\/|$)/.test(link.pathname)) {
      // Replace a special topdir name.
      result = link.protocol + '//' + link.host + '/' + topdir + '/' +
        link.pathname.replace(/\/[^\/]*(?:\/|$)/, '') + link.search + link.hash;
    }
  } else if (isPencilHost(global.location.hostname)) {
    // Proxy offdomain requests to avoid CORS issues.
    result = '/proxy/' + result;
  }
  return result;
}
// Creates an image url from a potentially short name.
function imgUrl(url) {
  if (/\//.test(url)) { return url; }
  url = '/img/' + url;
  if (isPencilHost(global.location.hostname)) { return url; }
  return '//pencilcode.net' + url;
}
// Retrieves the pencil code login cookie, if there is one.
function loginCookie() {
  if (!document.cookie) return null;
  var cookies = document.cookie.split(/;\s*/);
  for (var j = 0; j < cookies.length; ++j) {
    if (/^login=/.test(cookies[j])) {
      var val = unescape(cookies[j].substr(6)).split(':');
      if (val && val.length == 2) {
        return { user: val[0], key: val[1] };
      }
    }
  }
  return null;
}

// A map of url to {img: Image, queue: [{elem: elem, css: css, cb: cb}]}.
var stablyLoadedImages = {};

// setImageWithStableOrigin
//
// Changes the src of an <img> while keeping its transformOrigin
// at the same screen postition (by adjusting the transform).
// Because loading an image from a remote URL is an async operation
// that will move the center of an image at an indeterminate moment,
// this function loads the image in an off-screen objects first, and
// then once the image is loaded, it uses the loaded image to
// determine the natural dimensions; and then it sets these
// dimensions at the same time as setting the <img> src, and
// adjusts the transform according to any change in transformOrigin.
//
// @param elem is the <img> element whose src is to be set.
// @param url is the desried value of the src attribute.
// @param css is a dictionary of css props to set when the image is loaded.
// @param cb is an optional callback, called after the loading is done.
function setImageWithStableOrigin(elem, url, css, cb) {
  var record, urlobj = absoluteUrlObject(url);
  url = urlobj.href;
  // The data-loading attr will always reflect the last URL requested.
  elem.setAttribute('data-loading', url);
  if (url in stablyLoadedImages) {
    // Already requested this image?
    record = stablyLoadedImages[url];
    if (record.queue === null) {
      // If already complete, then flip the image right away.
      finishSet(record.img, elem, css, cb);
    } else {
      // If not yet complete, then add the target element to the queue.
      record.queue.push({elem: elem, css: css, cb: cb});
      // Pop the element to the right dimensions early if possible.
      resizeEarlyIfPossible(url, elem, css);
    }
  } else {
    // Set up a new image load.
    stablyLoadedImages[url] = record = {
      img: new Image(),
      queue: [{elem: elem, css: css, cb: cb}]
    };
    if (isPencilHost(urlobj.hostname)) {
      // When requesting through pencilcode, always make a
      // cross-origin request.
      record.img.crossOrigin = 'Anonymous';
    }
    // Pop the element to the right dimensions early if possible.
    resizeEarlyIfPossible(url, elem, css);
    // First set up the onload callback, then start loading.
    afterImageLoadOrError(record.img, url, function() {
      var j, queue = record.queue;
      record.queue = null;
      if (queue) {
        // Finish every element that hasn't yet been finished.
        for (j = 0; j < queue.length; ++j) {
          finishSet(record.img, queue[j].elem, queue[j].css, queue[j].cb);
        }
      }
    });
  }
  // This is the second step, done after the async load is complete:
  // the parameter "loaded" contains the fully loaded Image.
  function finishSet(loaded, elem, css, cb) {
    // Only flip the src if the last requested image is the same as
    // the one we have now finished loading: otherwise, there has been
    // some subsequent load that has now superceded ours.
    if (elem.getAttribute('data-loading') == loaded.src) {
      elem.removeAttribute('data-loading');
      applyLoadedImage(loaded, elem, css);
    }
    // Call the callback, if any.
    if (cb) {
      cb();
    }
  }
}

function afterImageLoadOrError(img, url, fn) {
  if (url == null) { url = img.src; }
  // If already loaded, then just call fn.
  if (url == img.src && (!url || img.complete)) {
    fn();
    return;
  }
  // Otherwise, set up listeners and wait.
  var timeout = null;
  function poll(e) {
    // If we get a load or error event, notice img.complete
    // or see that the src was changed, we're done here.
    if (e || img.complete || img.src != url) {
      img.removeEventListener('load', poll);
      img.removeEventListener('error', poll);
      clearTimeout(timeout);
      fn();
    } else {
      // Otherwise, continue waiting and also polling.
      timeout = setTimeout(poll, 100);
    }
  }
  img.addEventListener('load', poll);
  img.addEventListener('error', poll);
  img.src = url;
  poll();
}

// In the special case of loading a data: URL onto an element
// where we also have an explicit css width and height to apply,
// we go ahead and synchronously apply the CSS properties even if
// the URL isn't yet marked as loaded.  This is needed to allow
// "wear pointer" to act synchronously for tests even though
// PhantomJS asynchronously loads data: url images.  (Note that
// Chrome syncrhonously loads data: url images, so this is a
// dead code path on Chrome.)
function resizeEarlyIfPossible(url, elem, css) {
  if (/^data:/.test(url) && css.width && css.height) {
    applyLoadedImage(null, elem, css);
  }
}

function applyLoadedImage(loaded, elem, css) {
  // Read the element's origin before setting the image src.
  var oldOrigin = readTransformOrigin(elem),
      sel = $(elem),
      isCanvas = (elem.tagName == 'CANVAS'),
      ctx;
  if (!isCanvas) {
    // Set the image to a 1x1 transparent GIF, and clear the transform origin.
    // (This "reset" code was original added in an effort to avoid browser
    // bugs, but it is not clear if it is still needed.)
    elem.src = 'data:image/gif;base64,R0lGODlhAQABAIAAA' +
               'AAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';
  }
  sel.css({
    backgroundImage: 'none',
    height: '',
    width: '',
    turtleHull: '',
    transformOrigin: ''
  });
  if (loaded) {
    // Now set the source, and then apply any css requested.
    if (loaded.tagName == 'VIDEO') {
      elem.width = $(loaded).width();
      elem.height = $(loaded).height();
      if (isCanvas) {
        ctx = elem.getContext('2d');
        ctx.clearRect(0, 0, elem.width, elem.height);
        ctx.drawImage(loaded, 0, 0, loaded.videoWidth, loaded.videoHeight,
            0, 0, elem.width, elem.height);
      }
    } else {
      elem.width = loaded.width;
      elem.height = loaded.height;
      if (!isCanvas) {
        elem.src = loaded.src;
      } else if (loaded.width > 0 && loaded.height > 0) {
        try {
          ctx = elem.getContext('2d');
          ctx.clearRect(0, 0, loaded.width, loaded.height);
          ctx.drawImage(loaded, 0, 0);
        } catch (e) { }
      }
    }
  }
  if (css) {
    sel.css(css);
  }
  var newOrigin = readTransformOrigin(elem);
  if (loaded && !css.turtleHull) {
    try {
      var hull = transparentHull(loaded);
      scalePolygon(hull,
            parseFloat(sel.css('width')) / loaded.width,
            parseFloat(sel.css('height')) / loaded.height,
            -newOrigin[0], -newOrigin[1]);
      sel.css('turtleHull', hull);
    } catch (e) {
      // Do not do this if the image can't be loaded.
    }
  }
  moveToPreserveOrigin(elem, oldOrigin, newOrigin);
}

function moveToPreserveOrigin(elem, oldOrigin, newOrigin) {
  var sel = $(elem);
  if (!sel.hasClass('turtle')) return;
  // If there was a change, then translate the element to keep the origin
  // in the same location on the screen.
  if (newOrigin[0] != oldOrigin[0] || newOrigin[1] != oldOrigin[1]) {
    if (sel.css('position') == 'absolute' &&
        /px$/.test(sel.css('left')) && /px$/.test(sel.css('top'))) {
      // Do the translation using absolute positioning, if absolute.
      sel.css('left',
          parseFloat(sel.css('left')) + oldOrigin[0] - newOrigin[0]);
      sel.css('top',
          parseFloat(sel.css('top')) + oldOrigin[1] - newOrigin[1]);
    } else {
      // Do the translation using CSS transforms otherwise.
      var ts = readTurtleTransform(elem, true);
      ts.tx += oldOrigin[0] - newOrigin[0];
      ts.ty += oldOrigin[1] - newOrigin[1];
      elem.style[transform] = writeTurtleTransform(ts);
    }
  }
}

function withinOrNot(obj, within, distance, x, y) {
  var sel, elem, gbcr, pos, d2;
  if (x === undefined && y === undefined) {
    sel = $(distance);
    if (!sel.length) { return []; }
    elem = sel[0];
    gbcr = getPageGbcr(elem);
    if (polyMatchesGbcr(getCornersInPageCoordinates(elem), gbcr)) {
      return obj.filter(function() {
        var thisgbcr = getPageGbcr(this);
        return within === (gbcrEncloses(gbcr, thisgbcr) ||
            (!isDisjointGbcr(gbcr, thisgbcr) && $(this).inside(elem)));
      });
    } else {
      return obj.filter(function() {
        return within === $(this).inside(elem);
      });
    }
  }
  if ($.isNumeric(x) && $.isNumeric(y)) {
    pos = [x, y];
  } else {
    pos = x;
  }
  if ($.isArray(pos)) {
    // [x, y]: local coordinates.
    pos = convertLocalXyToPageCoordinates(obj[0] || document.body, [pos])[0];
  }
  if (distance === 'touch') {
    if (isPageCoordinate(pos)) {
      return obj.filter(function() {
        return within === $(this).touches(pos);
      });
    } else {
      sel = $(pos);
      gbcr = getPageGbcr(sel[0]);
      if (polyMatchesGbcr(getCornersInPageCoordinates(sel[0]), gbcr)) {
        return obj.filter(function() {
          var thisgbcr = getPageGbcr(this);
          // !isDisjoint test assumes gbcr is tight.
          return within === (!isDisjointGbcr(gbcr, thisgbcr) &&
            (gbcrEncloses(gbcr, thisgbcr) || sel.touches(this)));
        });
      } else {
        return obj.filter(function() {
          return within === sel.touches(this);
        });
      }
    }
  }
  d2 = distance * distance;
  return obj.filter(function() {
    var gbcr = getPageGbcr(this);
    if (isGbcrOutside(pos, d2, gbcr)) { return !within; }
    if (isGbcrInside(pos, d2, gbcr)) { return within; }
    var thispos = getCenterInPageCoordinates(this),
        dx = pos.pageX - thispos.pageX,
        dy = pos.pageY - thispos.pageY;
    return within === (dx * dx + dy * dy <= d2);
  });
}

//////////////////////////////////////////////////////////////////////////
// JQUERY SUBCLASSING
// Classes to allow jQuery to be subclassed.
//////////////////////////////////////////////////////////////////////////

// Sprite extends the jQuery object prototype.
var Sprite = (function(_super) {
  __extends(Sprite, _super);

  function Sprite(selector, context) {
    this.constructor = jQuery;
    this.constructor.prototype = Object.getPrototypeOf(this);
    if (!selector || typeof(selector) == 'string' ||
        $.isPlainObject(selector) || typeof(selector) == 'number') {
      // Use hatchone to create an element.
      selector = hatchone(selector, context, '256x256');
    }
    jQuery.fn.init.call(this, selector, context, rootjQuery);
  }

  Sprite.prototype.pushStack = function() {
    var count, ret, same;
    ret = jQuery.fn.pushStack.apply(this, arguments);
    count = ret.length;
    same = count === this.length;
    while (same && count--) {
      same = same && this[count] === ret[count];
    }
    if (same) {
      return this;
    } else {
      return ret;
    }
  };

  return Sprite;

})(jQuery.fn.init);

// Pencil extends Sprite, and is invisible and fast by default.
var Pencil = (function(_super) {
  __extends(Pencil, _super);

  function Pencil(canvas) {
    // A pencil draws on a canvas.  Allow a selector or element.
    if (canvas && canvas.jquery && $.isFunction(canvas.canvas)) {
      canvas = canvas.canvas();
    }
    if (canvas && (canvas.tagName != 'CANVAS' ||
        typeof canvas.getContext != 'function')) {
      canvas = $(canvas).filter('canvas').get(0);
    }
    if (!canvas || canvas.tagName != 'CANVAS' ||
        typeof canvas.getContext != 'function') {
      canvas = null;
    }
    // The pencil is a sprite that just defaults to zero size.
    var context = canvas ? canvas.parentElement : null;
    var settings = { width: 0, height: 0, color: 'transparent' };
    Pencil.__super__.constructor.call(this, settings, context);
    // Set the pencil to hidden, infinite speed,
    // and drawing on the specifed canvas.
    this.each(function() {
      var state = getTurtleData(this);
      state.speed = Infinity;
      state.drawOnCanvas = canvas;
      this.style.display = 'none';
      if (canvas) {
        this.style[transform] = writeTurtleTransform(
            readTurtleTransform(canvas, true));
      }
    });
  }

  return Pencil;

})(Sprite);


// Turtle extends Sprite, and draws a turtle by default.
var Turtle = (function(_super) {
  __extends(Turtle, _super);

  function Turtle(arg, context) {
    // The turtle is a sprite that just defaults to the turtle shape.
    Turtle.__super__.constructor.call(this, hatchone(arg, context, 'turtle'));
  }

  return Turtle;

})(Sprite);

// Webcam extends Sprite, and draws a live video camera by default.
var Webcam = (function(_super) {
  __extends(Webcam, _super);
  function Webcam(opts, context) {
    var attrs = "", hassrc = false, hasautoplay = false, hasdims = false;
    if ($.isPlainObject(opts)) {
      for (var key in opts) {
        attrs += ' ' + key + '="' + escapeHtml(opts[key]) + '"';
      }
      hassrc = ('src' in opts);
      hasautoplay = ('autoplay' in opts);
      hasdims = ('width' in opts || 'height' in opts);
      if (hasdims && !('height' in opts)) {
        attrs += ' height=' + Math.round(opts.width * 3/4);
      }
      if (hasdims && !('width' in opts)) {
        attrs += ' width=' + Math.round(opts.height * 4/3);
      }
    }
    if (!hasautoplay) {
      attrs += ' autoplay';
    }
    if (!hasdims) {
      attrs += ' width=320 height=240';
    }
    Webcam.__super__.constructor.call(this, '<video' + attrs + '>');
    if (!hassrc) {
      this.capture();
    }
  }
  Webcam.prototype.capture = function() {
    return this.queue(function(next) {
      var v = this,
          getUserMedia = navigator.getUserMedia ||
                         navigator.webkitGetUserMedia ||
                         navigator.mozGetUserMedia ||
                         navigator.msGetUserMedia;
      if (!getUserMedia) { next(); return; }
      getUserMedia.call(navigator, {video: true}, function(stream) {
        if (stream) {
          var state = getTurtleData(v), k = ('' + Math.random()).substr(2);
          if (state.stream) {
            state.stream.stop();
          }
          state.stream = stream;
          $(v).on('play.capture' + k, function() {
            $(v).off('play.capture' + k);
            next();
          });
          v.src = global.URL.createObjectURL(stream);
        }
      }, function() {
        next();
      });
    });
  };
  // Disconnects the media stream.
  Webcam.prototype.cut = function() {
    return this.plan(function() {
      var state = this.data('turtleData');
      if (state.stream) {
        state.stream.stop();
        state.stream = null;
      }
      this.attr('src', '');
    });
  };
  return Webcam;
})(Sprite);

// Piano extends Sprite, and draws a piano keyboard by default.
var Piano = (function(_super) {
  __extends(Piano, _super);
  // The piano constructor accepts an options object that can have:
  //   keys: the number of keys.  (this is the default property)
  //   color: the color of the white keys.
  //   blackColor: the color of the black keys.
  //   lineColor: the color of the key outlines.
  //   width: the overall keyboard pixel width.
  //   height: the overall keyboard pixel height.
  //   lineWidth: the outline line width.
  //   lowest: the lowest key number (as a midi number).
  //   highest: the highest key number (as a midi number).
  //   timbre: an Instrument timbre object or string.
  // Any subset of these properties may be supplied, and reasonable
  // defaults are chosen for everything else.  For example,
  // new Piano(88) will create a standard 88-key Piano keyboard.
  function Piano(options) {
    var aspect, defwidth, extra, firstwhite, height, width, lastwhite,
        numwhite = null, self = this, key, timbre, lowest, highest;
    options = parseOptionString(options, 'keys');
    // Convert options lowest and highest to midi numbers, if given.
    lowest = Instrument.pitchToMidi(options.lowest);
    highest = Instrument.pitchToMidi(options.highest);
    // The purpose of the logic in the constructor below is to calculate
    // reasonable defaults for the geometry of the keyboard given any
    // subset of options.  The geometric measurments go into _geom.
    var geom = this._geom = {}
    geom.lineWidth = ('lineWidth' in options) ? options.lineWidth : 1.5;
    geom.color = ('color' in options) ? options.color : 'white';
    geom.blackColor = ('blackColor' in options) ? options.blackColor : 'black';
    geom.lineColor = ('lineColor' in options) ? options.lineColor : 'black';
    // The extra pixel amount added to the bottom and right to take into
    // account the line width.
    extra = Math.ceil(geom.lineWidth);
    // Compute dimensions: first, default to 422 pixels wide and 100 tall.
    defwidth = 422;
    aspect = 4.2;
    // But if a key count is specified, default width to keep each white
    // key (i.e., 7/12ths of the keys) about 5:1 tall and the total
    // keyboard area about 42000 pixels.
    if (lowest != null && highest != null) {
      numwhite = wcp(highest) - wcp(lowest) + 1;
    } else if ('keys' in options) {
      numwhite = Math.ceil(options.keys / 12 * 7);
    }
    if (numwhite) {
      aspect = numwhite / 5;
      defwidth = Math.sqrt(42000 * aspect) + extra;
    }
    // If not specified explicitly, compute width from height, or if that
    // was not specified either, use the default width.
    width = ('width' in options) ? options.width : ('height' in options) ?
        Math.round((options.height - extra) * aspect + extra): defwidth;
    // Compute the height from width if not specified.
    height = ('height' in options) ? options.height :
        Math.round((width - extra) / aspect + extra);
    // If no key count, then come up with one based on geometry.
    if (!numwhite) {
      numwhite =
          Math.max(1, Math.round((width - extra) / (height - extra) * 5));
    }
    // Default rightmost white key by centering at F above middle C, up to C8.
    // For example, for 36 keys, there are 21 white keys, and the last
    // white key is 42 + (21 - 1) / 2 = 52, the B ten white keys above the F.
    lastwhite = Math.min(wcp(108), Math.ceil(42 + (numwhite - 1) / 2));
    // If the highest midi key is not specified, then use the default.
    geom.highest =
      (highest != null) ? highest :
      (lowest != null && 'keys' in options) ? lowest + options.keys - 1 :
      (lowest != null) ? mcp(wcp(lowest) + numwhite - 1) :
      mcp(lastwhite);
    // If the lowest midi key is not specified, then pick one.
    geom.lowest =
      (lowest != null) ? lowest :
      ('keys' in options) ? geom.highest - options.keys + 1 :
      Math.min(geom.highest, mcp(wcp(geom.highest) - numwhite + 1));
    // Final geometry computation.
    firstwhite = wcp(geom.lowest);
    lastwhite = wcp(geom.highest);
    // If highest is a black key, add the space of an extra white key.
    if (isblackkey(geom.highest)) { lastwhite += 1; }
    numwhite = lastwhite - firstwhite + 1;
    // Width and height of a single white key.
    geom.kw = (width - extra) / numwhite;
    geom.kh = (('height' in options) ? options.height - extra : geom.kw * 5) +
      (extra - geom.lineWidth); // Add roundoff to align with sprite bottom.
    // Width and height of a single black key.
    geom.bkw = geom.kw * 4 / 7;
    geom.bkh = geom.kh * 3 / 5;
    // Pixel offsets for centering the keyboard.
    geom.halfex = extra / 2;
    geom.leftpx = firstwhite * geom.kw;
    geom.rightpx = (lastwhite + 1) * geom.kw;
    // The top width of a C key and an F key (making space for black keys).
    geom.ckw = (3 * geom.kw - 2 * geom.bkw) / 3;
    geom.fkw = (4 * geom.kw - 3 * geom.bkw) / 4;
    Piano.__super__.constructor.call(this, {
      width: Math.ceil(geom.rightpx - geom.leftpx + extra),
      height: Math.ceil(geom.kh + extra)
    });
    // The following is a simplistic wavetable simulation of a Piano sound.
    if ('timbre' in options) {
      timbre = options.timbre;
    } else {
      // Allow timbre to be passed directly as options params.
      for (key in Instrument.defaultTimbre) {
        if (key in options) {
          if (!timbre) { timbre = {}; }
          timbre[key] = options[key];
        }
      }
    }
    if (!timbre) { timbre = 'piano'; }
    this.css({ turtleTimbre: timbre });
    // Hook up events.
    this.on('noteon', function(e) {
      self.drawkey(e.midi, keycolor(e.midi));
    });
    this.on('noteoff', function(e) {
      self.drawkey(e.midi);
    });
    this.draw();
    return this;
  }

  // Draws the key a midi number n, using the provided fill color
  // (defaults to white or black as appropriate).
  Piano.prototype.drawkey = function(n, fillcolor) {
    var ctx, geom = this._geom;
    if (!((geom.lowest <= n && n <= geom.highest))) {
      return;
    }
    if (fillcolor == null) {
      if (isblackkey(n)) {
        fillcolor = geom.blackColor;
      } else {
        fillcolor = geom.color;
      }
    }
    ctx = this.canvas().getContext('2d');
    ctx.save();
    ctx.beginPath();
    keyoutline(ctx, geom, n);
    ctx.fillStyle = fillcolor;
    ctx.strokeStyle = geom.lineColor;
    ctx.lineWidth = geom.lineWidth;
    ctx.fill();
    ctx.stroke();
    return ctx.restore();
  };

  // Draws every key on the keyboard.
  Piano.prototype.draw = function() {
    for (var n = this._geom.lowest; n <= this._geom.highest; ++n) {
      this.drawkey(n);
    }
  };

  var colors12 = [
    '#db4437', // C  red
    '#ff5722', // C# orange
    '#f4b400', // D  orange yellow
    '#ffeb3b', // D# yellow
    '#cddc39', // E  lime
    '#0f9d58', // F  green
    '#00bcd4', // F# teal
    '#03a9f4', // G  light blue
    '#4285f4', // G# blue
    '#673ab7', // A  deep purple
    '#9c27b0', // A# purple
    '#e91e63'  // B  pink
  ];

  // Picks a "noteon" color for a midi key number.
  function keycolor(n) {
    return colors12[(n % 12 + 12) % 12];
  };

  // Converts a midi number to a white key position (black keys round left).
  function wcp(n) {
    return Math.floor((n + 7) / 12 * 7);
  };

  // Converts from a white key position to a midi number.
  function mcp(n) {
    return Math.ceil(n / 7 * 12) - 7;
  };

  // True if midi #n is a black key.
  function isblackkey(n) {
    return keyshape(n) >= 8;
  }

  // Returns 1-8 for white keys CDEFGAB, and 9-12 for black keys C#D#F#G#A#.
  function keyshape(n) {
    return [1, 8, 2, 9, 3, 4, 10, 5, 11, 6, 12, 7][((n % 12) + 12) % 12];
  };

  // Given a 2d drawing context and geometry params, outlines midi key #n.
  function keyoutline(ctx, geom, n) {
    var ks, lcx, leftmost, rcx, rightmost, startx, starty;
    // The lower-left corner of the nearest (rounding left) white key.
    startx = geom.halfex + geom.kw * wcp(n) - geom.leftpx;
    starty = geom.halfex;
    // Compute the 12 cases of key shapes, plus special cases for the ends.
    ks = keyshape(n);
    leftmost = n === geom.lowest;
    rightmost = n === geom.highest;
    // White keys can have two cutouts: lcx is the x measurement of the
    // left cutout and rcx is the x measurement of the right cutout.
    lcx = 0;
    rcx = 0;
    switch (ks) {
      case 1:  // C
        rcx = geom.kw - geom.ckw;
        break;
      case 2:  // D
        rcx = lcx = (geom.kw - geom.ckw) / 2;
        break;
      case 3:  // E
        lcx = geom.kw - geom.ckw;
        break;
      case 4:  // F
        rcx = geom.kw - geom.fkw;
        break;
      case 5:  // G
        lcx = geom.fkw + geom.bkw - geom.kw;
        rcx = 2 * geom.kw - 2 * geom.fkw - geom.bkw;
        break;
      case 6:  // A
        lcx = 2 * geom.kw - 2 * geom.fkw - geom.bkw;
        rcx = geom.fkw + geom.bkw - geom.kw;
        break;
      case 7:  // B
        lcx = geom.kw - geom.fkw;
        break;
      case 8:  // C#
        startx += geom.ckw;
        break;
      case 9:  // D#
        startx += 2 * geom.ckw + geom.bkw - geom.kw;
        break;
      case 10: // F#
        startx += geom.fkw;
        break;
      case 11: // G#
        startx += 2 * geom.fkw + geom.bkw - geom.kw;
        break;
      case 12: // A#
        startx += 3 * geom.fkw + 2 * geom.bkw - 2 * geom.kw;
    }
    if (leftmost) {
      lcx = 0;
    }
    if (rightmost) {
      rcx = 0;
    }
    if (isblackkey(n)) {
      // A black key is always a rectangle.  Startx is computed above.
      ctx.moveTo(startx, starty + geom.bkh);
      ctx.lineTo(startx + geom.bkw, starty + geom.bkh);
      ctx.lineTo(startx + geom.bkw, starty);
      ctx.lineTo(startx, starty);
      return ctx.closePath();
    } else {
      // A white keys is a rectangle with two cutouts.
      ctx.moveTo(startx, starty + geom.kh);
      ctx.lineTo(startx + geom.kw, starty + geom.kh);
      ctx.lineTo(startx + geom.kw, starty + geom.bkh);
      ctx.lineTo(startx + geom.kw - rcx, starty + geom.bkh);
      ctx.lineTo(startx + geom.kw - rcx, starty);
      ctx.lineTo(startx + lcx, starty);
      ctx.lineTo(startx + lcx, starty + geom.bkh);
      ctx.lineTo(startx, starty + geom.bkh);
      return ctx.closePath();
    }
  };

  return Piano;

})(Sprite);

//////////////////////////////////////////////////////////////////////////
// KEYBOARD HANDLING
// Implementation of the "pressed" function
//////////////////////////////////////////////////////////////////////////

var focusTakenOnce = false;
function focusWindowIfFirst() {
  if (focusTakenOnce) return;
  focusTakenOnce = true;
  try {
    // If we are in a frame with access to a parent with an activeElement,
    // then try to blur it (as is common inside the pencilcode IDE).
    global.parent.document.activeElement.blur();
  } catch (e) {}
  global.focus();
}

// Construction of keyCode names.
var keyCodeName = (function() {
  var ua = typeof global !== 'undefined' ? global.navigator.userAgent : '',
      isOSX = /OS X/.test(ua),
      isOpera = /Opera/.test(ua),
      maybeFirefox = !/like Gecko/.test(ua) && !isOpera,
      pressedState = {},
      preventable = 'contextmenu',
      events = 'mousedown mouseup keydown keyup blur ' + preventable,
      keyCodeName = {
    0:  'null',
    1:  'mouse1',
    2:  'mouse2',
    3:  'break',
    4:  'mouse3',
    5:  'mouse4',
    6:  'mouse5',
    8:  'backspace',
    9:  'tab',
    12: 'clear',
    13: 'enter',
    16: 'shift',
    17: 'control',
    18: 'alt',
    19: 'pause',
    20: 'capslock',
    21: 'hangulmode',
    23: 'junjamode',
    24: 'finalmode',
    25: 'kanjimode',
    27: 'escape',
    28: 'convert',
    29: 'nonconvert',
    30: 'accept',
    31: 'modechange',
    27: 'escape',
    32: 'space',
    33: 'pageup',
    34: 'pagedown',
    35: 'end',
    36: 'home',
    37: 'left',
    38: 'up',
    39: 'right',
    40: 'down',
    41: 'select',
    42: 'print',
    43: 'execute',
    44: 'snapshot',
    45: 'insert',
    46: 'delete',
    47: 'help',
  // no one handles meta-left and right properly, so we coerce into one.
    91: 'meta',  // meta-left
    92: 'meta',  // meta-right
  // chrome,opera,safari all report this for meta-right (osx mbp).
    93: isOSX ? 'meta' : 'menu',
    95: 'sleep',
    106: 'numpad*',
    107: 'numpad+',
    108: 'numpadenter',
    109: 'numpad-',
    110: 'numpad.',
    111: 'numpad/',
    144: 'numlock',
    145: 'scrolllock',
    160: 'shiftleft',
    161: 'shiftright',
    162: 'controlleft',
    163: 'controlright',
    164: 'altleft',
    165: 'altright',
    166: 'browserback',
    167: 'browserforward',
    168: 'browserrefresh',
    169: 'browserstop',
    170: 'browsersearch',
    171: 'browserfavorites',
    172: 'browserhome',
    // ff/osx reports 'volume-mute' for '-'
    173: isOSX && maybeFirefox ? '-' : 'volumemute',
    174: 'volumedown',
    175: 'volumeup',
    176: 'mediatracknext',
    177: 'mediatrackprev',
    178: 'mediastop',
    179: 'mediaplaypause',
    180: 'launchmail',
    181: 'launchmediaplayer',
    182: 'launchapp1',
    183: 'launchapp2',
    186: ';',
    187: '=',
    188: ',',
    189: '-',
    190: '.',
    191: '/',
    192: '`',
    219: '[',
    220: '\\',
    221: ']',
    222: "'",
    223: 'meta',
    224: 'meta',      // firefox reports meta here.
    226: 'altgraph',
    229: 'process',
    231: isOpera ? '`' : 'unicode',
    246: 'attention',
    247: 'crsel',
    248: 'exsel',
    249: 'eraseeof',
    250: 'play',
    251: 'zoom',
    252: 'noname',
    253: 'pa1',
    254: 'clear'
  };
  // :-@, 0-9, a-z(lowercased)
  for (var i = 48; i < 91; ++i) {
    keyCodeName[i] = String.fromCharCode(i).toLowerCase();
  }
  // num-0-9 numeric keypad
  for (i = 96; i < 106; ++i) {
    keyCodeName[i] = 'numpad' +  (i - 96);
  }
  // f1-f24
  for (i = 112; i < 136; ++i) {
    keyCodeName[i] = 'f' + (i-111);
  }
  return keyCodeName;
})();

var pressedKey = (function() {
  // Listener for keyboard, mouse, and focus events that updates pressedState.
  function makeEventListener(mouse, down) {
    return (function(event) {
      var name, simplified, which = event.which;
      if (mouse) {
        name = 'mouse' + which;
      } else {
        // For testability, support whichSynth when which is zero, because
        // it is impossible to simulate .which on phantom.
        if (!which && event.whichSynth) { which = event.whichSynth; }
        name = keyCodeName[which];
        if (which >= 160 && which <= 165) {
          // For "shift left", also trigger "shift"; same for control and alt.
          updatePressedState(name.replace(/(?:left|right)$/, ''), down);
        }
      }
      updatePressedState(name, down);
    });
  };
  var eventMap = {
    'mousedown': makeEventListener(1, 1),
    'mouseup': makeEventListener(1, 0),
    'keydown': makeEventListener(0, 1),
    'keyup': makeEventListener(0, 0),
    'blur': resetPressedState
  };
  // The pressedState map just has an entry for each pressed key.
  // Unpressing a key will delete the actual key from the map.
  var pressedState = {};
  function updatePressedState(name, down) {
    if (name != null) {
      if (!down) {
        delete pressedState[name];
      } else {
        pressedState[name] = true;
      }
    }
  }
  // The state map is reset by clearing every member.
  function resetPressedState() {
    for (var key in pressedState) {
      delete pressedState[key];
    }
  }
  // The pressed listener can be turned on and off using pressed.enable(flag).
  function enablePressListener(turnon) {
    resetPressedState();
    for (var name in eventMap) {
      if (turnon) {
        global.addEventListener(name, eventMap[name], true);
      } else {
        global.removeEventListener(name, eventMap[name]);
      }
    }
  }
  // All pressed keys known can be listed using pressed.list().
  function listPressedKeys() {
    var result = [], key;
    for (key in pressedState) {
      if (pressedState[key]) { result.push(key); }
    }
    return result;
  }
  // The pressed function just polls the given keyname.
  function pressed(keyname) {
    focusWindowIfFirst();
    if (keyname) {
      // Canonical names are lowercase and have no spaces.
      keyname = keyname.replace(/\s/g, '').toLowerCase();
      if (pressedState[keyname]) return true;
      return false;
    } else {
      return listPressedKeys();
    }
  }
  pressed.enable = enablePressListener;
  pressed.list = listPressedKeys;
  return pressed;
})();


//////////////////////////////////////////////////////////////////////////
// JQUERY EVENT ENHANCEMENT
//  - Keyboard events get the .key property.
//  - Keyboard event listening with a string first (data) arg
//    automatically filter out events that don't match the keyname.
//  - Mouse events get .x and .y (center-up) if there is a turtle field.
//  - If a turtle in the field is listening to mouse events, unhandled
//    body mouse events are manually forwarded to turtles.
//////////////////////////////////////////////////////////////////////////

function addEventHook(hookobj, field, defobj, name, fn) {
  var names = name.split(/\s+/);
  for (var j = 0; j < names.length; ++j) {
    name = names[j];
    var hooks = hookobj[name];
    if (!hooks) {
      hooks = hookobj[name] = $.extend({}, defobj);
    }
    if (typeof hooks[field] != 'function') {
      hooks[field] = fn;
    } else if (hooks[field] != fn) {
      // Multiple event hooks just listed in an array.
      if (hooks[field].hooklist) {
        if (hooks[field].hooklist.indexOf(fn) < 0) {
          hooks[field].hooklist.push(fn);
        }
      } else {
        (function() {
          var hooklist = [hooks[field], fn];
          (hooks[field] = function(event, original) {
            var current = event;
            for (var j = 0; j < hooklist.length; ++j) {
              current = hooklist[j](current, original) || current;
            }
            return current;
          }).hooklist = hooklist;
        })();
      }
    }
  }
}

function mouseFilterHook(event, original) {
  if (globalDrawing.field && 'pageX' in event && 'pageY' in event) {
    var origin = $(globalDrawing.field).offset();
    if (origin) {
      event.x = event.pageX - origin.left;
      event.y = origin.top - event.pageY;
    }
  }
  return event;
}

function mouseSetupHook(data, ns, fn) {
  if (globalDrawing.field && !globalDrawing.fieldMouse &&
      this.parentElement === globalDrawing.field ||
      /(?:^|\s)turtle(?:\s|$)/.test(this.class)) {
    globalDrawing.fieldMouse = true;
    forwardBodyMouseEventsIfNeeded();
  }
  return false;
}

function forwardBodyMouseEventsIfNeeded() {
  if (globalDrawing.fieldHook) return;
  if (globalDrawing.surface && globalDrawing.fieldMouse) {
    globalDrawing.fieldHook = true;
    setTimeout(function() {
      // TODO: check both globalDrawing.surface and
      // globalDrawing.turtleMouseListener
      $('body').on('click.turtle dblclick.turtle ' +
        'mouseup.turtle mousedown.turtle mousemove.turtle', function(e) {
        if (e.target === this && !e.isTrigger) {
          // Only forward events directly on the body that (geometrically)
          // touch a turtle directly within the turtlefield.
          var warn = $.turtle.nowarn;
          $.turtle.nowarn = true;
          var sel = $(globalDrawing.surface)
              .find('.turtle,.turtlelabel').within('touch', e).eq(0);
          $.turtle.nowarn = warn;
          if (sel.length === 1) {
            // Erase portions of the event that are wrong for the turtle.
            e.target = null;
            e.relatedTarget = null;
            e.fromElement = null;
            e.toElement = null;
            sel.trigger(e);
            return false;
          }
        }
      });
    }, 0);
  }
}

function addMouseEventHooks() {
  var hookedEvents = 'mousedown mouseup mousemove click dblclick';
  addEventHook($.event.fixHooks, 'filter', $.event.mouseHooks,
       hookedEvents, mouseFilterHook);
  addEventHook($.event.special, 'setup', {}, hookedEvents, mouseSetupHook);
}

function keyFilterHook(event, original) {
  var which = event.which;
  if (!which) {
    which = (original || event.originalEvent).whichSynth;
  }
  var name = keyCodeName[which];
  if (!name && which) {
    name = String.fromCharCode(which);
  }
  event.key = name;
  return event;
}

// Add .key to each keyboard event.
function keypressFilterHook(event, original) {
  if (event.charCode != null) {
    event.key = String.fromCharCode(event.charCode);
  }
}

// Intercept on('keydown/keyup/keypress')
function keyAddHook(handleObj) {
  if (typeof(handleObj.data) != 'string') return;
  var choices = handleObj.data.replace(/\s/g, '').toLowerCase().split(',');
  var original = handleObj.handler;
  var wrapped = function(event) {
    if (choices.indexOf(event.key) < 0) return;
    return original.apply(this, arguments);
  }
  if (original.guid) { wrapped.guid = original.guid; }
  handleObj.handler = wrapped;
}

function addKeyEventHooks() {
  // Add the "key" field to keydown and keyup events - this uses
  // the lowercase key names listed in the pressedKey utility.
  addEventHook($.event.fixHooks, 'filter', $.event.keyHooks,
      'keydown keyup', keyFilterHook);
  // Add "key" to keypress also.  This is just the unicode character
  // corresponding to event.charCode.
  addEventHook($.event.fixHooks, 'filter', $.event.keyHooks,
      'keypress', keypressFilterHook);
  // Finally, add special forms for the keyup/keydown/keypress events
  // where the first argument can be the comma-separated name of keys
  // to target (instead of just data)
  addEventHook($.event.special, 'add', {},
      'keydown keyup keypress', keyAddHook);
}

//////////////////////////////////////////////////////////////////////////
// WEB AUDIO SUPPORT
// Definition of play("ABC") - uses ABC music note syntax.
//////////////////////////////////////////////////////////////////////////


// jQuery CSS hook for turtleTimbre property.
function makeTimbreHook() {
  return {
    get: function(elem, computed, extra) {
      return printOptionAsString(getTurtleInstrument(elem).getTimbre());
    },
    set: function(elem, value) {
      getTurtleInstrument(elem).setTimbre(parseOptionString(value, 'wave'));
    }
  };
}

// jQuery CSS hook for turtleVolume property.
function makeVolumeHook() {
  return {
    get: function(elem, computed, extra) {
      return getTurtleInstrument(elem).getVolume();
    },
    set: function(elem, value) {
      getTurtleInstrument(elem).setVolume(parseFloat(value));
    }
  };
}

// Every HTML element gets an instrument.  This creates and returns it.
function getTurtleInstrument(elem) {
  var state = getTurtleData(elem);
  if (state.instrument) {
    return state.instrument;
  }
  state.instrument = new Instrument("piano");
  // Hook up noteon and noteoff events.
  var selector = $(elem);
  state.instrument.on('noteon', function(r) {
    var event = $.Event('noteon');
    event.midi = r.midi;
    selector.trigger(event);
  });
  state.instrument.on('noteoff', function(r) {
    var event = $.Event('noteoff');
    event.midi = r.midi;
    selector.trigger(event);
  });
  return state.instrument;
}

// In addition, threre is a global instrument.  This funcion returns it.
var global_instrument = null;
function getGlobalInstrument() {
  if (!global_instrument) {
    global_instrument = new Instrument();
  }
  return global_instrument;
}

// Beginning of musical.js copy

// Tests for the presence of HTML5 Web Audio (or webkit's version).
function isAudioPresent() {
  return !!(global.AudioContext || global.webkitAudioContext);
}

// All our audio funnels through the same AudioContext with a
// DynamicsCompressorNode used as the main output, to compress the
// dynamic range of all audio.  getAudioTop sets this up.
function getAudioTop() {
  if (getAudioTop.audioTop) { return getAudioTop.audioTop; }
  if (!isAudioPresent()) {
    return null;
  }
  var ac = new (global.AudioContext || global.webkitAudioContext);
  getAudioTop.audioTop = {
    ac: ac,
    wavetable: makeWavetable(ac),
    out: null,
    currentStart: null
  };
  resetAudio();
  return getAudioTop.audioTop;
}

// When audio needs to be interrupted globally (e.g., when you press the
// stop button in the IDE), resetAudio does the job.
function resetAudio() {
  if (getAudioTop.audioTop) {
    var atop = getAudioTop.audioTop;
    // Disconnect the top-level node and make a new one.
    if (atop.out) {
      atop.out.disconnect();
      atop.out = null;
      atop.currentStart = null;
    }
    // If resetting due to interrupt after AudioContext closed, this can fail.
    try {
      var dcn = atop.ac.createDynamicsCompressor();
      dcn.ratio = 16;
      dcn.attack = 0.0005;
      dcn.connect(atop.ac.destination);
      atop.out = dcn;
    } catch (e) {
      getAudioTop.audioTop = null;
    }
  }
}

// For precise scheduling of future notes, the AudioContext currentTime is
// cached and is held constant until the script releases to the event loop.
function audioCurrentStartTime() {
  var atop = getAudioTop();
  if (atop.currentStart != null) {
    return atop.currentStart;
  }
  // A delay could be added below to introduce a universal delay in
  // all beginning sounds (without skewing durations for scheduled
  // sequences).
  atop.currentStart = Math.max(0.25, atop.ac.currentTime /* + 0.0 delay */);
  setTimeout(function() { atop.currentStart = null; }, 0);
  return atop.currentStart;
}

// Converts a midi note number to a frequency in Hz.
function midiToFrequency(midi) {
  return 440 * Math.pow(2, (midi - 69) / 12);
}
// Some constants.
var noteNum =
    {C:0,D:2,E:4,F:5,G:7,A:9,B:11,c:12,d:14,e:16,f:17,g:19,a:21,b:23};
var accSym =
    { '^':1, '': 0, '=':0, '_':-1 };
var noteName =
    ['C', '^C', 'D', '_E', 'E', 'F', '^F', 'G', '_A', 'A', '_B', 'B',
     'c', '^c', 'd', '_e', 'e', 'f', '^f', 'g', '_a', 'a', '_b', 'b'];
// Converts a frequency in Hz to the closest midi number.
function frequencyToMidi(freq) {
  return Math.round(69 + Math.log(freq / 440) * 12 / Math.LN2);
}
// Converts an ABC pitch (such as "^G,,") to a midi note number.
function pitchToMidi(pitch) {
  var m = /^(\^+|_+|=|)([A-Ga-g])([,']*)$/.exec(pitch);
  if (!m) { return null; }
  var octave = m[3].replace(/,/g, '').length - m[3].replace(/'/g, '').length;
  var semitone =
      noteNum[m[2]] + accSym[m[1].charAt(0)] * m[1].length + 12 * octave;
  return semitone + 60; // 60 = midi code middle "C".
}
// Converts a midi number to an ABC notation pitch.
function midiToPitch(midi) {
  var index = ((midi - 72) % 12);
  if (midi > 60 || index != 0) { index += 12; }
  var octaves = Math.round((midi - index - 60) / 12),
      result = noteName[index];
  while (octaves != 0) {
    result += octaves > 0 ? "'" : ",";
    octaves += octaves > 0 ? -1 : 1;
  }
  return result;
}
// Converts an ABC pitch to a frequency in Hz.
function pitchToFrequency(pitch) {
  return midiToFrequency(pitchToMidi(pitch));
}

// All further details of audio handling are encapsulated in the Instrument
// class, which knows how to synthesize a basic timbre; how to play and
// schedule a tone; and how to parse and sequence a song written in ABC
// notation.
var Instrument = (function() {
  // The constructor accepts a timbre string or object, specifying
  // its default sound.  The main mechanisms in Instrument are for handling
  // sequencing of a (potentially large) set of notes over a (potentially
  // long) period of time.  The overall strategy:
  //
  //                       Events:      'noteon'        'noteoff'
  //                                      |               |
  // tone()-(quick tones)->| _startSet -->| _finishSet -->| _cleanupSet -->|
  //   \                   |  /           | Playing tones | Done tones     |
  //    \---- _queue ------|-/                                             |
  //      of future tones  |3 secs ahead sent to WebAudio, removed when done
  //
  // The reason for this queuing is to reduce the complexity of the
  // node graph sent to WebAudio: at any time, WebAudio is only
  // responsible for about 2 seconds of music.  If a graph with too
  // too many nodes is sent to WebAudio at once, output distorts badly.
  function Instrument(options) {
    this._atop = getAudioTop();    // Audio context.
    this._timbre = makeTimbre(options, this._atop); // The instrument's timbre.
    this._queue = [];              // A queue of future tones to play.
    this._minQueueTime = Infinity; // The earliest time in _queue.
    this._maxScheduledTime = 0;    // The latest time in _queue.
    this._unsortedQueue = false;   // True if _queue is unsorted.
    this._startSet = [];           // Unstarted tones already sent to WebAudio.
    this._finishSet = {};          // Started tones playing in WebAudio.
    this._cleanupSet = [];         // Tones waiting for cleanup.
    this._callbackSet = [];        // A set of scheduled callbacks.
    this._handlers = {};           // 'noteon' and 'noteoff' handlers.
    this._now = null;              // A cached current-time value.
    if (isAudioPresent()) {
      this.silence();              // Initializes top-level audio node.
    }
  }

  Instrument.timeOffset = 0.0625;// Seconds to delay all audiable timing.
  Instrument.dequeueTime = 0.5;  // Seconds before an event to reexamine queue.
  Instrument.bufferSecs = 2;     // Seconds ahead to put notes in WebAudio.
  Instrument.toneLength = 1;     // Default duration of a tone.
  Instrument.cleanupDelay = 0.1; // Silent time before disconnecting nodes.

  // Sets the default timbre for the instrument.  See defaultTimbre.
  Instrument.prototype.setTimbre = function(t) {
    this._timbre = makeTimbre(t, this._atop);     // Saves a copy.
  };

  // Returns the default timbre for the instrument as an object.
  Instrument.prototype.getTimbre = function(t) {
    return makeTimbre(this._timbre, this._atop);  // Makes a copy.
  };

  // Sets the overall volume for the instrument immediately.
  Instrument.prototype.setVolume = function(v) {
    // Without an audio system, volume cannot be set.
    if (!this._out) { return; }
    if (!isNaN(v)) {
      this._out.gain.value = v;
    }
  };

  // Sets the overall volume for the instrument.
  Instrument.prototype.getVolume = function(v) {
    // Without an audio system, volume is stuck at zero.
    if (!this._out) { return 0.0; }
    return this._out.gain.value;
  };

  // Silences the instrument immediately by reinitializing the audio
  // graph for this instrument and emptying or flushing all queues in the
  // scheduler.  Carefully notifies all notes that have started but not
  // yet finished, and sequences that are awaiting scheduled callbacks.
  // Does not notify notes that have not yet started.
  Instrument.prototype.silence = function() {
    var j, finished, callbacks, initvolume = 1;

    // Clear future notes.
    this._queue.length = 0;
    this._minQueueTime = Infinity;
    this._maxScheduledTime = 0;

    // Don't notify notes that haven't started yet.
    this._startSet.length = 0;

    // Flush finish callbacks that are promised.
    finished = this._finishSet;
    this._finishSet = {};

    // Flush one-time callacks that are promised.
    callbacks = this._callbackSet;
    this._callbackSet = [];

    // Disconnect the audio graph for this instrument.
    if (this._out) {
      this._out.disconnect();
      initvolume = this._out.gain.value;
    }

    // Reinitialize the audio graph: all audio for the instrument
    // multiplexes through a single gain node with a master volume.
    this._atop = getAudioTop();
    this._out = this._atop.ac.createGain();
    this._out.gain.value = initvolume;
    this._out.connect(this._atop.out);

    // As a last step, call all promised notifications.
    for (j in finished) { this._trigger('noteoff', finished[j]); }
    for (j = 0; j < callbacks.length; ++j) { callbacks[j].callback(); }
  };

  // Future notes are scheduled relative to now(), which provides
  // access to audioCurrentStartTime(), a time that holds steady
  // until the script releases to the event loop.  When _now is
  // non-null, it indicates that scheduling is already in progress.
  // The timer-driven _doPoll function clears the cached _now.
  Instrument.prototype.now = function() {
    if (this._now != null) {
      return this._now;
    }
    this._startPollTimer(true);  // passing (true) sets this._now.
    return this._now;
  };

  // Register an event handler.  Done without jQuery to reduce dependencies.
  Instrument.prototype.on = function(eventname, cb) {
    if (!this._handlers.hasOwnProperty(eventname)) {
      this._handlers[eventname] = [];
    }
    this._handlers[eventname].push(cb);
  };

  // Unregister an event handler.  Done without jQuery to reduce dependencies.
  Instrument.prototype.off = function(eventname, cb) {
    if (this._handlers.hasOwnProperty(eventname)) {
      if (!cb) {
        this._handlers[eventname] = [];
      } else {
        var j, hunt = this._handlers[eventname];
        for (j = 0; j < hunt.length; ++j) {
          if (hunt[j] === cb) {
            hunt.splice(j, 1);
            j -= 1;
          }
        }
      }
    }
  };

  // Trigger an event, notifying any registered handlers.
  Instrument.prototype._trigger = function(eventname, record) {
    var cb = this._handlers[eventname], j;
    if (!cb) { return; }
    if (cb.length == 1) {
      // Special, common case of one handler: no copy needed.
      cb[0](record);
      return;
    }
    // Copy the array of callbacks before iterating, because the
    // main this._handlers copy could be changed by a handler.
    // You get notified if-and-only-if you are registered
    // at the starting moment of _trigger.
    cb = cb.slice();
    for (j = 0; j < cb.length; ++j) {
      cb[j](record);
    }
  };

  // Tells the WebAudio API to play a tone (now or soon).  The passed
  // record specifies a start time and release time, an ADSR envelope,
  // and other timbre parameters.  This function sets up a WebAudio
  // node graph for the tone generators and filters for the tone.
  Instrument.prototype._makeSound = function(record) {
    var timbre = record.timbre || this._timbre,
        starttime = record.time + Instrument.timeOffset,
        releasetime = starttime + record.duration,
        attacktime = Math.min(releasetime, starttime + timbre.attack),
        decaytime = timbre.decay *
            Math.pow(440 / record.frequency, timbre.decayfollow),
        decaystarttime = attacktime,
        stoptime = releasetime + timbre.release,
        doubled = timbre.detune && timbre.detune != 1.0,
        amp = timbre.gain * record.velocity * (doubled ? 0.5 : 1.0),
        ac = this._atop.ac,
        g, f, o, o2, pwave, k, wf, bwf;
    // Only hook up tone generators if it is an audible sound.
    if (record.duration > 0 && record.velocity > 0) {
      g = ac.createGain();
      g.gain.setValueAtTime(0, starttime);
      g.gain.linearRampToValueAtTime(amp, attacktime);
      // For the beginning of the decay, use linearRampToValue instead
      // of setTargetAtTime, because it avoids http://crbug.com/254942.
      while (decaystarttime < attacktime + 1/32 &&
             decaystarttime + 1/256 < releasetime) {
        // Just trace out the curve in increments of 1/256 sec
        // for up to 1/32 seconds.
        decaystarttime += 1/256;
        g.gain.linearRampToValueAtTime(
            amp * (timbre.sustain + (1 - timbre.sustain) *
                Math.exp((attacktime - decaystarttime) / decaytime)),
            decaystarttime);
      }
      // For the rest of the decay, use setTargetAtTime.
      g.gain.setTargetAtTime(amp * timbre.sustain,
          decaystarttime, decaytime);
      // Then at release time, mark the value and ramp to zero.
      g.gain.setValueAtTime(amp * (timbre.sustain + (1 - timbre.sustain) *
          Math.exp((attacktime - releasetime) / decaytime)), releasetime);
      g.gain.linearRampToValueAtTime(0, stoptime);
      g.connect(this._out);
      // Hook up a low-pass filter if cutoff is specified.
      if ((!timbre.cutoff && !timbre.cutfollow) || timbre.cutoff == Infinity) {
        f = g;
      } else {
        // Apply the cutoff frequency adjusted using cutfollow.
        f = ac.createBiquadFilter();
        f.frequency.value =
            timbre.cutoff + record.frequency * timbre.cutfollow;
        f.Q.value = timbre.resonance;
        f.connect(g);
      }
      // Hook up the main oscillator.
      o = makeOscillator(this._atop, timbre.wave, record.frequency);
      o.connect(f);
      o.start(starttime);
      o.stop(stoptime);
      // Hook up a detuned oscillator.
      if (doubled) {
        o2 = makeOscillator(
            this._atop, timbre.wave, record.frequency * timbre.detune);
        o2.connect(f);
        o2.start(starttime);
        o2.stop(stoptime);
      }
      // Store nodes in the record so that they can be modified
      // in case the tone is truncated later.
      record.gainNode = g;
      record.oscillators = [o];
      if (doubled) { record.oscillators.push(o2); }
      record.cleanuptime = stoptime;
    } else {
      // Inaudible sounds are scheduled: their purpose is to truncate
      // audible tones at the same pitch.  But duration is set to zero
      // so that they are cleaned up quickly.
      record.duration = 0;
    }
    this._startSet.push(record);
  };
  // Truncates a sound previously scheduled by _makeSound by using
  // cancelScheduledValues and directly ramping down to zero.
  // Can only be used to shorten a sound.
  Instrument.prototype._truncateSound = function(record, truncatetime) {
    if (truncatetime < record.time + record.duration) {
      record.duration = Math.max(0, truncatetime - record.time);
      if (record.gainNode) {
        var timbre = record.timbre || this._timbre,
            starttime = record.time + Instrument.timeOffset,
            releasetime = truncatetime + Instrument.timeOffset,
            attacktime = Math.min(releasetime, starttime + timbre.attack),
            decaytime = timbre.decay *
                Math.pow(440 / record.frequency, timbre.decayfollow),
            stoptime = releasetime + timbre.release,
            cleanuptime = stoptime + Instrument.cleanupDelay,
            doubled = timbre.detune && timbre.detune != 1.0,
            amp = timbre.gain * record.velocity * (doubled ? 0.5 : 1.0),
            j, g = record.gainNode;
        // Cancel any envelope points after the new releasetime.
        g.gain.cancelScheduledValues(releasetime);
        if (releasetime <= starttime) {
          // Release before start?  Totally silence the note.
          g.gain.setValueAtTime(0, releasetime);
        } else if (releasetime <= attacktime) {
          // Release before attack is done?  Interrupt ramp up.
          g.gain.linearRampToValueAtTime(
            amp * (releasetime - starttime) / (attacktime - starttime),
            releasetime);
        } else {
          // Release during decay?  Interrupt decay down.
          g.gain.setValueAtTime(amp * (timbre.sustain + (1 - timbre.sustain) *
            Math.exp((attacktime - releasetime) / decaytime)), releasetime);
        }
        // Then ramp down to zero according to record.release.
        g.gain.linearRampToValueAtTime(0, stoptime);
        // After stoptime, stop the oscillators.  This is necessary to
        // eliminate extra work for WebAudio for no-longer-audible notes.
        if (record.oscillators) {
          for (j = 0; j < record.oscillators.length; ++j) {
            record.oscillators[j].stop(stoptime);
          }
        }
        // Schedule disconnect.
        record.cleanuptime = cleanuptime;
      }
    }
  };
  // The core scheduling loop is managed by Instrument._doPoll.  It reads
  // the audiocontext's current time and pushes tone records from one
  // stage to the next.
  //
  // 1. The first stage is the _queue, which has tones that have not
  //    yet been given to WebAudio. This loop scans _queue to find
  //    notes that need to begin in the next few seconds; then it
  //    sends those to WebAduio and moves them to _startSet. Because
  //    scheduled songs can be long, _queue can be large.
  //
  // 2. Second is _startSet, which has tones that have been given to
  //    WebAudio, but whose start times have not yet elapsed. When
  //    the time advances past the start time of a record, a 'noteon'
  //    notification is fired for the tone, and it is moved to
  //    _finishSet.
  //
  // 3. _finishSet represents the notes that are currently sounding.
  //    The programming model for Instrument is that only one tone of
  //    a specific frequency may be played at once within a Instrument,
  //    so only one tone of a given frequency may exist in _finishSet
  //    at once.  When there is a conflict, the sooner-to-end-note
  //    is truncated.
  //
  // 4. After a note is released, it may have a litle release time
  //    (depending on timbre.release), after which the nodes can
  //    be totally disconnected and cleaned up.  _cleanupSet holds
  //    notes for which we are awaiting cleanup.
  Instrument.prototype._doPoll = function() {
    this._pollTimer = null;
    this._now = null;
    if (interrupted) {
      this.silence();
      return;
    }
    // The shortest time we can delay is 1 / 1000 secs, so if an event
    // is within the next 0.5 ms, now is the closest moment, and we go
    // ahead and process it.
    var instant = this._atop.ac.currentTime + (1 / 2000),
        callbacks = [],
        j, work, when, freq, record, conflict, save, cb;
    // Schedule a batch of notes
    if (this._minQueueTime - instant <= Instrument.bufferSecs) {
      if (this._unsortedQueue) {
        this._queue.sort(function(a, b) {
          if (a.time != b.time) { return a.time - b.time; }
          if (a.duration != b.duration) { return a.duration - b.duration; }
          return a.frequency - b.frequency;
        });
        this._unsortedQueue = false;
      }
      for (j = 0; j < this._queue.length; ++j) {
        if (this._queue[j].time - instant > Instrument.bufferSecs) { break; }
      }
      if (j > 0) {
        work = this._queue.splice(0, j);
        for (j = 0; j < work.length; ++j) {
          this._makeSound(work[j]);
        }
        this._minQueueTime =
          (this._queue.length > 0) ? this._queue[0].time : Infinity;
      }
    }
    // Disconnect notes from the cleanup set.
    for (j = 0; j < this._cleanupSet.length; ++j) {
      record = this._cleanupSet[j];
      if (record.cleanuptime < instant) {
        if (record.gainNode) {
          // This explicit disconnect is needed or else Chrome's WebAudio
          // starts getting overloaded after a couple thousand notes.
          record.gainNode.disconnect();
          record.gainNode = null;
        }
        this._cleanupSet.splice(j, 1);
        j -= 1;
      }
    }
    // Notify about any notes finishing.
    for (freq in this._finishSet) {
      record = this._finishSet[freq];
      when = record.time + record.duration;
      if (when <= instant) {
        callbacks.push({
          order: [when, 0],
          f: this._trigger, t: this, a: ['noteoff', record]});
        if (record.cleanuptime != Infinity) {
          this._cleanupSet.push(record);
        }
        delete this._finishSet[freq];
      }
    }
    // Call any specific one-time callbacks that were registered.
    for (j = 0; j < this._callbackSet.length; ++j) {
      cb = this._callbackSet[j];
      when = cb.time;
      if (when <= instant) {
        callbacks.push({
          order: [when, 1],
          f: cb.callback, t: null, a: []});
        this._callbackSet.splice(j, 1);
        j -= 1;
      }
    }
    // Notify about any notes starting.
    for (j = 0; j < this._startSet.length; ++j) {
      if (this._startSet[j].time <= instant) {
        save = record = this._startSet[j];
        freq = record.frequency;
        conflict = null;
        if (this._finishSet.hasOwnProperty(freq)) {
          // If there is already a note at the same frequency playing,
          // then release the one that starts first, immediately.
          conflict = this._finishSet[freq];
          if (conflict.time < record.time || (conflict.time == record.time &&
              conflict.duration < record.duration)) {
            // Our new sound conflicts with an old one: end the old one
            // and notify immediately of its noteoff event.
            this._truncateSound(conflict, record.time);
            callbacks.push({
              order: [record.time, 0],
              f: this._trigger, t: this, a: ['noteoff', conflict]});
            delete this._finishSet[freq];
          } else {
            // A conflict from the future has already scheduled,
            // so our own note shouldn't sound.  Truncate ourselves
            // immediately, and suppress our own noteon and noteoff.
            this._truncateSound(record, conflict.time);
            conflict = record;
          }
        }
        this._startSet.splice(j, 1);
        j -= 1;
        if (record.duration > 0 && record.velocity > 0 &&
            conflict !== record) {
          this._finishSet[freq] = record;
          callbacks.push({
            order: [record.time, 2],
            f: this._trigger, t: this, a: ['noteon', record]});
        }
      }
    }
    // Schedule the next _doPoll.
    this._startPollTimer();

    // Sort callbacks according to the "order" tuple, so earlier events
    // are notified first.
    callbacks.sort(function(a, b) {
      if (a.order[0] != b.order[0]) { return a.order[0] - b.order[0]; }
      // tiebreak by notifying 'noteoff' first and 'noteon' last.
      return a.order[1] - b.order[1];
    });
    // At the end, call all the callbacks without depending on "this" state.
    for (j = 0; j < callbacks.length; ++j) {
      cb = callbacks[j];
      cb.f.apply(cb.t, cb.a);
    }
  };
  // Schedules the next _doPoll call by examining times in the various
  // sets and determining the soonest event that needs _doPoll processing.
  Instrument.prototype._startPollTimer = function(setnow) {
    // If we have already done a "setnow", then pollTimer is zero-timeout
    // and cannot be faster.
    if (this._pollTimer && this._now != null) {
      return;
    }
    var self = this,
        poll = function() { self._doPoll(); },
        earliest = Infinity, j, delay;
    if (this._pollTimer) {
      // Clear any old timer
      clearTimeout(this._pollTimer);
      this._pollTimer = null;
    }
    if (setnow) {
      // When scheduling tones, cache _now and keep a zero-timeout poll.
      // _now will be cleared the next time we execute _doPoll.
      this._now = audioCurrentStartTime();
      this._pollTimer = setTimeout(poll, 0);
      return;
    }
    // Timer due to notes starting: wake up for 'noteon' notification.
    for (j = 0; j < this._startSet.length; ++j) {
      earliest = Math.min(earliest, this._startSet[j].time);
    }
    // Timer due to notes finishing: wake up for 'noteoff' notification.
    for (j in this._finishSet) {
      earliest = Math.min(
        earliest, this._finishSet[j].time + this._finishSet[j].duration);
    }
    // Timer due to scheduled callback.
    for (j = 0; j < this._callbackSet.length; ++j) {
      earliest = Math.min(earliest, this._callbackSet[j].time);
    }
    // Timer due to cleanup: add a second to give some time to batch up.
    if (this._cleanupSet.length > 0) {
      earliest = Math.min(earliest, this._cleanupSet[0].cleanuptime + 1);
    }
    // Timer due to sequencer events: subtract a little time to stay ahead.
    earliest = Math.min(
       earliest, this._minQueueTime - Instrument.dequeueTime);

    delay = Math.max(0.001, earliest - this._atop.ac.currentTime);

    // If there are no future events, then we do not need a timer.
    if (isNaN(delay) || delay == Infinity) { return; }

    // Use the Javascript timer to wake up at the right moment.
    this._pollTimer = setTimeout(poll, Math.round(delay * 1000));
  };

  // The low-level tone function.
  Instrument.prototype.tone =
  function(pitch, duration, velocity, delay, timbre, origin) {
    // If audio is not present, this is a no-op.
    if (!this._atop) { return; }

    // Called with an object instead of listed args.
    if (typeof(pitch) == 'object') {
      if (velocity == null) velocity = pitch.velocity;
      if (duration == null) duration = pitch.duration;
      if (delay == null) delay = pitch.delay;
      if (timbre == null) timbre = pitch.timbre;
      if (origin == null) origin = pitch.origin;
      pitch = pitch.pitch;
    }

    // Convert pitch from various formats to Hz frequency and a midi num.
    var midi, frequency;
    if (!pitch) { pitch = 'C'; }
    if (isNaN(pitch)) {
      midi = pitchToMidi(pitch);
      frequency = midiToFrequency(midi);
    } else {
      frequency = Number(pitch);
      if (frequency < 0) {
        midi = -frequency;
        frequency = midiToFrequency(midi);
      } else {
        midi = frequencyToMidi(frequency);
      }
    }

    if (!timbre) {
      timbre = this._timbre;
    }
    // If there is a custom timbre, validate and copy it.
    if (timbre !== this._timbre) {
      var given = timbre, key;
      timbre = {}
      for (key in defaultTimbre) {
        if (key in given) {
          timbre[key] = given[key];
        } else {
          timbre[key] = defaultTimbre[key];
        }
      }
    }

    // Create the record for a tone.
    var ac = this._atop.ac,
        now = this.now(),
        time = now + (delay || 0),
        record = {
          time: time,
          on: false,
          frequency: frequency,
          midi: midi,
          velocity: (velocity == null ? 1 : velocity),
          duration: (duration == null ? Instrument.toneLength : duration),
          timbre: timbre,
          instrument: this,
          gainNode: null,
          oscillators: null,
          cleanuptime: Infinity,
          origin: origin             // save the origin of the tone for visible feedback
        };

    if (time < now + Instrument.bufferSecs) {
      // The tone starts soon!  Give it directly to WebAudio.
      this._makeSound(record);
    } else {
      // The tone is later: queue it.
      if (!this._unsortedQueue && this._queue.length &&
          time < this._queue[this._queue.length -1].time) {
        this._unsortedQueue = true;
      }
      this._queue.push(record);
      this._minQueueTime = Math.min(this._minQueueTime, record.time);
    }
  };
  // The low-level callback scheduling method.
  Instrument.prototype.schedule = function(delay, callback) {
    this._callbackSet.push({ time: this.now() + delay, callback: callback });
  };
  // The high-level sequencing method.
  Instrument.prototype.play = function(abcstring) {
    var args = Array.prototype.slice.call(arguments),
        done = null,
        opts = {}, subfile,
        abcfile, argindex, tempo, timbre, k, delay, maxdelay = 0, attenuate,
        voicename, stems, ni, vn, j, stem, note, beatsecs, secs, v, files = [];
    // Look for continuation as last argument.
    if (args.length && 'function' == typeof(args[args.length - 1])) {
      done = args.pop();
    }
    if (!this._atop) {
      if (done) { done(); }
      return;
    }
    // Look for options as first object.
    argindex = 0;
    if ('object' == typeof(args[0])) {
      // Copy own properties into an options object.
      for (k in args[0]) if (args[0].hasOwnProperty(k)) {
        opts[k] = args[0][k];
      }
      argindex = 1;
      // If a song is supplied by options object, process it.
      if (opts.song) {
        args.push(opts.song);
      }
    }
    // Parse any number of ABC files as input.
    for (; argindex < args.length; ++argindex) {
      // Handle splitting of ABC subfiles at X: lines.
      subfile = args[argindex].split(/\n(?=X:)/);
      for (k = 0; k < subfile.length; ++k) {
        abcfile = parseABCFile(subfile[k]);
        if (!abcfile) continue;
        // Take tempo markings from the first file, and share them.
        if (!opts.tempo && abcfile.tempo) {
          opts.tempo = abcfile.tempo;
          if (abcfile.unitbeat) {
            opts.tempo *= abcfile.unitbeat / (abcfile.unitnote || 1);
          }
        }
        // Ignore files without songs.
        if (!abcfile.voice) continue;
        files.push(abcfile);
      }
    }
    // Default tempo to 120 if nothing else is specified.
    if (!opts.tempo) { opts.tempo = 120; }
    // Default volume to 1 if nothing is specified.
    if (opts.volume == null) { opts.volume = 1; }
    beatsecs = 60.0 / opts.tempo;
    // Schedule all notes from all the files.
    for (k = 0; k < files.length; ++k) {
      abcfile = files[k];
      // Each file can have multiple voices (e.g., left and right hands)
      for (vn in abcfile.voice) {
        // Each voice could have a separate timbre.
        timbre = makeTimbre(opts.timbre || abcfile.voice[vn].timbre ||
           abcfile.timbre || this._timbre, this._atop);
        // Each voice has a series of stems (notes or chords).
        stems = abcfile.voice[vn].stems;
        if (!stems) continue;
        // Starting at delay zero (now), schedule all tones.
        delay = 0;
        for (ni = 0; ni < stems.length; ++ni) {
          stem = stems[ni];
          // Attenuate chords to reduce clipping.
          attenuate = 1 / Math.sqrt(stem.notes.length);
          // Schedule every note inside a stem.
          for (j = 0; j < stem.notes.length; ++j) {
            note = stem.notes[j];
            if (note.holdover) {
              // Skip holdover notes from ties.
              continue;
            }
            secs = (note.time || stem.time) * beatsecs;
            if (stem.staccato) {
              // Shorten staccato notes.
              secs = Math.min(Math.min(secs, beatsecs / 16),
                  timbre.attack + timbre.decay);
            } else if (!note.slurred && secs >= 1/8) {
              // Separate unslurred notes by about a 30th of a second.
              secs -= 1/32;
            }
            v = (note.velocity || 1) * attenuate * opts.volume;
            // This is innsermost part of the inner loop!
            this.tone(                     // Play the tone:
              note.pitch,                  // at the given pitch
              secs,                        // for the given duration
              v,                           // with the given volume
              delay,                       // starting at the proper time
              timbre,                      // with the selected timbre
              note                         // the origin object for visual feedback
              );
          }
          delay += stem.time * beatsecs;   // Advance the sequenced time.
        }
        maxdelay = Math.max(delay, maxdelay);
      }
    }
    this._maxScheduledTime =
        Math.max(this._maxScheduledTime, this.now() + maxdelay);
    if (done) {
      // Schedule a "done" callback after all sequencing is complete.
      this.schedule(maxdelay, done);
    }
  };


  // The default sound is a square wave with a pretty quick decay to zero.
  var defaultTimbre = Instrument.defaultTimbre = {
    wave: 'square',   // Oscillator type.
    gain: 0.1,        // Overall gain at maximum attack.
    attack: 0.002,    // Attack time at the beginning of a tone.
    decay: 0.4,       // Rate of exponential decay after attack.
    decayfollow: 0,   // Amount of decay shortening for higher notes.
    sustain: 0,       // Portion of gain to sustain indefinitely.
    release: 0.1,     // Release time after a tone is done.
    cutoff: 0,        // Low-pass filter cutoff frequency.
    cutfollow: 0,     // Cutoff adjustment, a multiple of oscillator freq.
    resonance: 0,     // Low-pass filter resonance.
    detune: 0         // Detune factor for a second oscillator.
  };

  // Norrmalizes a timbre object by making a copy that has exactly
  // the right set of timbre fields, defaulting when needed.
  // A timbre can specify any of the fields of defaultTimbre; any
  // unspecified fields are treated as they are set in defaultTimbre.
  function makeTimbre(options, atop) {
    if (!options) {
      options = {};
    }
    if (typeof(options) == 'string') {
      // Abbreviation: name a wave to get a default timbre for that wave.
      options = { wave: options };
    }
    var result = {}, key,
        wt = atop && atop.wavetable && atop.wavetable[options.wave];
    for (key in defaultTimbre) {
      if (options.hasOwnProperty(key)) {
        result[key] = options[key];
      } else if (wt && wt.defs && wt.defs.hasOwnProperty(key)) {
        result[key] = wt.defs[key];
      } else{
        result[key] = defaultTimbre[key];
      }
    }
    return result;
  }

  function getWhiteNoiseBuf() {
    var ac = getAudioTop().ac,
      bufferSize = 2 * ac.sampleRate,
      whiteNoiseBuf = ac.createBuffer(1, bufferSize, ac.sampleRate),
      output = whiteNoiseBuf.getChannelData(0);
    for (var i = 0; i < bufferSize; i++) {
      output[i] = Math.random() * 2 - 1;
    }
    return whiteNoiseBuf;
  }

  // This utility function creates an oscillator at the given frequency
  // and the given wavename.  It supports lookups in a static wavetable,
  // defined right below.
  function makeOscillator(atop, wavename, freq) {
    if (wavename == 'noise') {
      var whiteNoise = atop.ac.createBufferSource();
      whiteNoise.buffer = getWhiteNoiseBuf();
      whiteNoise.loop = true;
      return whiteNoise;
    }
    var wavetable = atop.wavetable, o = atop.ac.createOscillator(),
        k, pwave, bwf, wf;
    try {
      if (wavetable.hasOwnProperty(wavename)) {
        // Use a customized wavetable.
        pwave = wavetable[wavename].wave;
        if (wavetable[wavename].freq) {
          bwf = 0;
          // Look for a higher-frequency variant.
          for (k in wavetable[wavename].freq) {
            wf = Number(k);
            if (freq > wf && wf > bwf) {
              bwf = wf;
              pwave = wavetable[wavename].freq[bwf];
            }
          }
        }
        if (!o.setPeriodicWave && o.setWaveTable) {
          // The old API name: Safari 7 still uses this.
          o.setWaveTable(pwave);
        } else {
          // The new API name.
          o.setPeriodicWave(pwave);
        }
      } else {
        o.type = wavename;
      }
    } catch(e) {
      if (window.console) { window.console.log(e); }
      // If unrecognized, just use square.
      // TODO: support "noise" or other wave shapes.
      o.type = 'square';
    }
    o.frequency.value = freq;
    return o;
  }

  // Accepts either an ABC pitch or a midi number and converts to midi.
  Instrument.pitchToMidi = function(n) {
    if (typeof(n) == 'string') { return pitchToMidi(n); }
    return n;
  }

  // Accepts either an ABC pitch or a midi number and converts to ABC pitch.
  Instrument.midiToPitch = function(n) {
    if (typeof(n) == 'number') { return midiToPitch(n); }
    return n;
  }

  return Instrument;
})();

// Parses an ABC file to an object with the following structure:
// {
//   X: value from the X: lines in header (\n separated for multiple values)
//   V: value from the V:myname lines that appear before K:
//   (etc): for all the one-letter header-names.
//   K: value from the K: lines in header.
//   tempo: Q: line parsed as beatsecs
//   timbre: ... I:timbre line as parsed by makeTimbre
//   voice: {
//     myname: { // voice with id "myname"
//       V: value from the V:myname lines (from the body)
//       stems: [...] as parsed by parseABCstems
//    }
//  }
// }
// ABC files are idiosyncratic to parse: the written specifications
// do not necessarily reflect the defacto standard implemented by
// ABC content on the web.  This implementation is designed to be
// practical, working on content as it appears on the web, and only
// using the written standard as a guideline.
var ABCheader = /^([A-Za-z]):\s*(.*)$/;
var ABCtoken = /(?:\[[A-Za-z]:[^\]]*\])|\s+|%[^\n]*|![^\s!:|\[\]]*!|\+[^+|!]*\+|[_<>@^]?"[^"]*"|\[|\]|>+|<+|(?:(?:\^+|_+|=|)[A-Ga-g](?:,+|'+|))|\(\d+(?::\d+){0,2}|\d*\/\d+|\d+\/?|\/+|[xzXZ]|\[?\|\]?|:?\|:?|::|./g;
function parseABCFile(str) {
  var lines = str.split('\n'),
      result = {},
      context = result, timbre,
      j, k, header, stems, key = {}, accent = { slurred: 0 }, voiceid, out;
  // ABC files are parsed one line at a time.
  for (j = 0; j < lines.length; ++j) {
    // First, check to see if the line is a header line.
    header = ABCheader.exec(lines[j]);
    if (header) {
      handleInformation(header[1], header[2].trim());
    } else if (/^\s*(?:%.*)?$/.test(lines[j])) {
      // Skip blank and comment lines.
      continue;
    } else {
      // Parse the notes.
      parseABCNotes(lines[j]);
    }
  }
  var infer = ['unitnote', 'unitbeat', 'tempo'];
  if (result.voice) {
    out = [];
    for (j in result.voice) {
      if (result.voice[j].stems && result.voice[j].stems.length) {
        // Calculate times for all the tied notes.  This happens at the end
        // because in principle, the first note of a song could be tied all
        // the way through to the last note.
        processTies(result.voice[j].stems);
        // Bring up inferred tempo values from voices if not specified
        // in the header.
        for (k = 0; k < infer.length; ++k) {
          if (!(infer[k] in result) && (infer[k] in result.voice[j])) {
            result[infer[k]] = result.voice[j][infer[k]];
          }
        }
        // Remove this internal state variable;
        delete result.voice[j].accent;
      } else {
        out.push(j);
      }
    }
    // Delete any voices that had no stems.
    for (j = 0; j < out.length; ++j) {
      delete result.voice[out[j]];
    }
  }
  return result;


  ////////////////////////////////////////////////////////////////////////
  // Parsing helper functions below.
  ////////////////////////////////////////////////////////////////////////


  // Processes header fields such as V: voice, which may appear at the
  // top of the ABC file, or in the ABC body in a [V:voice] directive.
  function handleInformation(field, value) {
    // The following headers are recognized and processed.
    switch(field) {
      case 'V':
        // A V: header switches voices if in the body.
        // If in the header, then it is just advisory.
        if (context !== result) {
          startVoiceContext(value.split(' ')[0]);
        }
        break;
      case 'M':
        parseMeter(value, context);
        break;
      case 'L':
        parseUnitNote(value, context);
        break;
      case 'Q':
        parseTempo(value, context);
        break;
    }
    // All headers (including unrecognized ones) are
    // just accumulated as properties. Repeated header
    // lines are accumulated as multiline properties.
    if (context.hasOwnProperty(field)) {
      context[field] += '\n' + value;
    } else {
      context[field] = value;
    }
    // The K header is special: it should be the last one
    // before the voices and notes begin.
    if (field == 'K') {
      key = keysig(value);
      if (context === result) {
        startVoiceContext(firstVoiceName());
      }
    }
  }

  // Shifts context to a voice with the given id given.  If no id
  // given, then just sticks with the current voice.  If the current
  // voice is unnamed and empty, renames the current voice.
  function startVoiceContext(id) {
    id = id || '';
    if (!id && context !== result) {
      return;
    }
    if (!result.voice) {
      result.voice = {};
    }
    if (result.voice.hasOwnProperty(id)) {
      // Resume a named voice.
      context = result.voice[id];
      accent = context.accent;
    } else {
      // Start a new voice.
      context = { id: id, accent: { slurred: 0 } };
      result.voice[id] = context;
      accent = context.accent;
    }
  }

  // For picking a default voice, looks for the first voice name.
  function firstVoiceName() {
    if (result.V) {
      return result.V.split(/\s+/)[0];
    } else {
      return '';
    }
  }

  // Parses a single line of ABC notes (i.e., not a header line).
  //
  // We process an ABC song stream by dividing it into tokens, each of
  // which is a pitch, duration, or special decoration symbol; then
  // we process each decoration individually, and we process each
  // stem as a group using parseStem.
  // The structure of a single ABC note is something like this:
  //
  // NOTE -> STACCATO? PITCH DURATION? TIE?
  //
  // I.e., it always has a pitch, and it is prefixed by some optional
  // decorations such as a (.) staccato marking, and it is suffixed by
  // an optional duration and an optional tie (-) marking.
  //
  // A stem is either a note or a bracketed series of notes, followed
  // by duration and tie.
  //
  // STEM -> NOTE   OR    '[' NOTE * ']' DURAITON? TIE?
  //
  // Then a song is just a sequence of stems interleaved with other
  // decorations such as dynamics markings and measure delimiters.
  function parseABCNotes(str) {
    var tokens = str.match(ABCtoken), parsed = null,
        index = 0, dotted = 0, beatlet = null, t;
    if (!tokens) {
      return null;
    }
    while (index < tokens.length) {
      // Ignore %comments and !markings!
      if (/^[\s%]/.test(tokens[index])) { index++; continue; }
      // Handle inline [X:...] information fields
      if (/^\[[A-Za-z]:[^\]]*\]$/.test(tokens[index])) {
        handleInformation(
          tokens[index].substring(1, 2),
          tokens[index].substring(3, tokens[index].length - 1).trim()
        );
        index++;
        continue;
      }
      // Handled dotted notation abbreviations.
      if (/</.test(tokens[index])) {
        dotted = -tokens[index++].length;
        continue;
      }
      if (/>/.test(tokens[index])) {
        dotted = tokens[index++].length;
        continue;
      }
      if (/^\(\d+(?::\d+)*/.test(tokens[index])) {
        beatlet = parseBeatlet(tokens[index++]);
        continue;
      }
      if (/^[!+].*[!+]$/.test(tokens[index])) {
        parseDecoration(tokens[index++], accent);
        continue;
      }
      if (/^.?".*"$/.test(tokens[index])) {
        // Ignore double-quoted tokens (chords and general text annotations).
        index++;
        continue;
      }
      if (/^[()]$/.test(tokens[index])) {
        if (tokens[index++] == '(') {
          accent.slurred += 1;
        } else {
          accent.slurred -= 1;
          if (accent.slurred <= 0) {
            accent.slurred = 0;
            if (context.stems && context.stems.length >= 1) {
              // The last notes in a slur are not slurred.
              slurStem(context.stems[context.stems.length - 1], false);
            }
          }
        }
        continue;
      }
      // Handle measure markings by clearing accidentals.
      if (/\|/.test(tokens[index])) {
        for (t in accent) {
          if (t.length == 1) {
            // Single-letter accent properties are note accidentals.
            delete accent[t];
          }
        }
        index++;
        continue;
      }
      parsed = parseStem(tokens, index, key, accent);
      // Skip unparsable bits
      if (parsed === null) {
        index++;
        continue;
      }
      // Process a parsed stem.
      if (beatlet) {
        scaleStem(parsed.stem, beatlet.time);
        beatlet.count -= 1;
        if (!beatlet.count) {
          beatlet = null;
        }
      }
      // If syncopated with > or < notation, shift part of a beat
      // between this stem and the previous one.
      if (dotted && context.stems && context.stems.length) {
        if (dotted > 0) {
          t = (1 - Math.pow(0.5, dotted)) * parsed.stem.time;
        } else {
          t = (Math.pow(0.5, -dotted) - 1) *
              context.stems[context.stems.length - 1].time;
        }
        syncopateStem(context.stems[context.stems.length - 1], t);
        syncopateStem(parsed.stem, -t);
      }
      dotted = 0;
      // Slur all the notes contained within a strem.
      if (accent.slurred) {
        slurStem(parsed.stem, true);
      }
      // Start a default voice if we're not in a voice yet.
      if (context === result) {
        startVoiceContext(firstVoiceName());
      }
      if (!('stems' in context)) { context.stems = []; }
      // Add the stem to the sequence of stems for this voice.
      context.stems.push(parsed.stem);
      // Advance the parsing index since a stem is multiple tokens.
      index = parsed.index;
    }
  }

  // Parse M: lines.  "3/4" is 3/4 time and "C" is 4/4 (common) time.
  function parseMeter(mline, beatinfo) {
    var d = /^C/.test(mline) ? 4/4 : durationToTime(mline);
    if (!d) { return; }
    if (!beatinfo.unitnote) {
      if (d < 0.75) {
        beatinfo.unitnote = 1/16;
      } else {
        beatinfo.unitnote = 1/8;
      }
    }
  }
  // Parse L: lines, e.g., "1/8".
  function parseUnitNote(lline, beatinfo) {
    var d = durationToTime(lline);
    if (!d) { return; }
    beatinfo.unitnote = d;
  }
  // Parse Q: line, e.g., "1/4=66".
  function parseTempo(qline, beatinfo) {
    var parts = qline.split(/\s+|=/), j, unit = null, tempo = null;
    for (j = 0; j < parts.length; ++j) {
      // It could be reversed, like "66=1/4", or just "120", so
      // determine what is going on by looking for a slash etc.
      if (parts[j].indexOf('/') >= 0 || /^[1-4]$/.test(parts[j])) {
        // The note-unit (e.g., 1/4).
        unit = unit || durationToTime(parts[j]);
      } else {
        // The tempo-number (e.g., 120)
        tempo = tempo || Number(parts[j]);
      }
    }
    if (unit) {
      beatinfo.unitbeat = unit;
    }
    if (tempo) {
      beatinfo.tempo = tempo;
    }
  }
  // Run through all the notes, adding up time for tied notes,
  // and marking notes that were held over with holdover = true.
  function processTies(stems) {
    var tied = {}, nextTied, j, k, note, firstNote;
    for (j = 0; j < stems.length; ++j) {
      nextTied = {};
      for (k = 0; k < stems[j].notes.length; ++k) {
        firstNote = note = stems[j].notes[k];
        if (tied.hasOwnProperty(note.pitch)) {  // Pitch was tied from before.
          firstNote = tied[note.pitch];   // Get the earliest note in the tie.
          firstNote.time += note.time;    // Extend its time.
          note.holdover = true;           // Silence this note as a holdover.
        }
        if (note.tie) {                   // This note is tied with the next.
          nextTied[note.pitch] = firstNote;  // Save it away.
        }
      }
      tied = nextTied;
    }
  }
  // Returns a map of A-G -> accidentals, according to the key signature.
  // When n is zero, there are no accidentals (e.g., C major or A minor).
  // When n is positive, there are n sharps (e.g., for G major, n = 1).
  // When n is negative, there are -n flats (e.g., for F major, n = -1).
  function accidentals(n) {
    var sharps = 'FCGDAEB',
        result = {}, j;
    if (!n) {
      return result;
    }
    if (n > 0) {  // Handle sharps.
      for (j = 0; j < n && j < 7; ++j) {
        result[sharps.charAt(j)] = '^';
      }
    } else {  // Flats are in the opposite order.
      for (j = 0; j > n && j > -7; --j) {
        result[sharps.charAt(6 + j)] = '_';
      }
    }
    return result;
  }
  // Decodes the key signature line (e.g., K: C#m) at the front of an ABC tune.
  // Supports the whole range of scale systems listed in the ABC spec.
  function keysig(keyname) {
    if (!keyname) { return {}; }
    var kkey, sigcodes = {
      // Major
      'c#':7, 'f#':6, 'b':5, 'e':4, 'a':3, 'd':2, 'g':1, 'c':0,
      'f':-1, 'bb':-2, 'eb':-3, 'ab':-4, 'db':-5, 'gb':-6, 'cb':-7,
      // Minor
      'a#m':7, 'd#m':6, 'g#m':5, 'c#m':4, 'f#m':3, 'bm':2, 'em':1, 'am':0,
      'dm':-1, 'gm':-2, 'cm':-3, 'fm':-4, 'bbm':-5, 'ebm':-6, 'abm':-7,
      // Mixolydian
      'g#mix':7, 'c#mix':6, 'f#mix':5, 'bmix':4, 'emix':3,
      'amix':2, 'dmix':1, 'gmix':0, 'cmix':-1, 'fmix':-2,
      'bbmix':-3, 'ebmix':-4, 'abmix':-5, 'dbmix':-6, 'gbmix':-7,
      // Dorian
      'd#dor':7, 'g#dor':6, 'c#dor':5, 'f#dor':4, 'bdor':3,
      'edor':2, 'ador':1, 'ddor':0, 'gdor':-1, 'cdor':-2,
      'fdor':-3, 'bbdor':-4, 'ebdor':-5, 'abdor':-6, 'dbdor':-7,
      // Phrygian
      'e#phr':7, 'a#phr':6, 'd#phr':5, 'g#phr':4, 'c#phr':3,
      'f#phr':2, 'bphr':1, 'ephr':0, 'aphr':-1, 'dphr':-2,
      'gphr':-3, 'cphr':-4, 'fphr':-5, 'bbphr':-6, 'ebphr':-7,
      // Lydian
      'f#lyd':7, 'blyd':6, 'elyd':5, 'alyd':4, 'dlyd':3,
      'glyd':2, 'clyd':1, 'flyd':0, 'bblyd':-1, 'eblyd':-2,
      'ablyd':-3, 'dblyd':-4, 'gblyd':-5, 'cblyd':-6, 'fblyd':-7,
      // Locrian
      'b#loc':7, 'e#loc':6, 'a#loc':5, 'd#loc':4, 'g#loc':3,
      'c#loc':2, 'f#loc':1, 'bloc':0, 'eloc':-1, 'aloc':-2,
      'dloc':-3, 'gloc':-4, 'cloc':-5, 'floc':-6, 'bbloc':-7
    };
    var k = keyname.replace(/\s+/g, '').toLowerCase().substr(0, 5);
    var scale = k.match(/maj|min|mix|dor|phr|lyd|loc|m/);
    if (scale) {
      if (scale == 'maj') {
        kkey = k.substr(0, scale.index);
      } else if (scale == 'min') {
        kkey = k.substr(0, scale.index + 1);
      } else {
        kkey = k.substr(0, scale.index + scale[0].length);
      }
    } else {
      kkey = /^[a-g][#b]?/.exec(k) || '';
    }
    var result = accidentals(sigcodes[kkey]);
    var extras = keyname.substr(kkey.length).match(/(_+|=|\^+)[a-g]/ig);
    if (extras) {
      for (var j = 0; j < extras.length; ++j) {
        var note = extras[j].charAt(extras[j].length - 1).toUpperCase();
        if (extras[j].charAt(0) == '=') {
          delete result[note];
        } else {
          result[note] = extras[j].substr(0, extras[j].length - 1);
        }
      }
    }
    return result;
  }
  // Additively adjusts the beats for a stem and the contained notes.
  function syncopateStem(stem, t) {
    var j, note, stemtime = stem.time, newtime = stemtime + t;
    stem.time = newtime;
    for (j = 0; j < stem.notes.length; ++j) {
      note = stem.notes[j];
      // Only adjust a note's duration if it matched the stem's duration.
      if (note.time == stemtime) { note.time = newtime; }
    }
  }
  // Marks everything in the stem with the slur attribute (or deletes it).
  function slurStem(stem, addSlur) {
    var j, note;
    for (j = 0; j < stem.notes.length; ++j) {
      note = stem.notes[j];
      if (addSlur) {
        note.slurred = true;
      } else if (note.slurred) {
        delete note.slurred;
      }
    }
  }
  // Scales the beats for a stem and the contained notes.
  function scaleStem(stem, s) {
    var j;
    stem.time *= s;
    for (j = 0; j < stem.notes.length; ++j) {
      stem.notes[j].time *= s;;
    }
  }
  // Parses notation of the form (3 or (5:2:10, which means to do
  // the following 3 notes in the space of 2 notes, or to do the following
  // 10 notes at the rate of 5 notes per 2 beats.
  function parseBeatlet(token) {
    var m = /^\((\d+)(?::(\d+)(?::(\d+))?)?$/.exec(token);
    if (!m) { return null; }
    var count = Number(m[1]),
        beats = Number(m[2]) || 2,
        duration = Number(m[3]) || count;
    return {
      time: beats / count,
      count: duration
    };
  }
  // Parse !ppp! markings.
  function parseDecoration(token, accent) {
    if (token.length < 2) { return; }
    token = token.substring(1, token.length - 1);
    switch (token) {
      case 'pppp': case 'ppp':
        accent.dynamics = 0.2; break;
      case 'pp':
        accent.dynamics = 0.4; break;
      case 'p':
        accent.dynamics = 0.6; break;
      case 'mp':
        accent.dynamics = 0.8; break;
      case 'mf':
        accent.dynamics = 1.0; break;
      case 'f':
        accent.dynamics = 1.2; break;
      case 'ff':
        accent.dynamics = 1.4; break;
      case 'fff': case 'ffff':
        accent.dynamics = 1.5; break;
    }
  }
  // Parses a stem, which may be a single note, or which may be
  // a chorded note.
  function parseStem(tokens, index, key, accent) {
    var notes = [],
        duration = '', staccato = false,
        noteDuration, noteTime, velocity,
        lastNote = null, minStemTime = Infinity, j;
    // A single staccato marking applies to the entire stem.
    if (index < tokens.length && '.' == tokens[index]) {
      staccato = true;
      index++;
    }
    if (index < tokens.length && tokens[index] == '[') {
      // Deal with [CEG] chorded notation.
      index++;
      // Scan notes within the chord.
      while (index < tokens.length) {
        // Ignore and space and %comments.
        if (/^[\s%]/.test(tokens[index])) {
          index++;
          continue;
        }
        if (/[A-Ga-g]/.test(tokens[index])) {
          // Grab a pitch.
          lastNote = {
            pitch: applyAccent(tokens[index++], key, accent),
            tie: false
          }
          lastNote.frequency = pitchToFrequency(lastNote.pitch);
          notes.push(lastNote);
        } else if (/[xzXZ]/.test(tokens[index])) {
          // Grab a rest.
          lastNote = null;
          index++;
        } else if ('.' == tokens[index]) {
          // A staccato mark applies to the entire stem.
          staccato = true;
          index++;
          continue;
        } else {
          // Stop parsing the stem if something is unrecognized.
          break;
        }
        // After a pitch or rest, look for a duration.
        if (index < tokens.length &&
            /^(?![\s%!]).*[\d\/]/.test(tokens[index])) {
          noteDuration = tokens[index++];
          noteTime = durationToTime(noteDuration);
        } else {
          noteDuration = '';
          noteTime = 1;
        }
        // If it's a note (not a rest), store the duration
        if (lastNote) {
          lastNote.duration = noteDuration;
          lastNote.time = noteTime;
        }
        // When a stem has more than one duration, use the shortest
        // one for timing. The standard says to pick the first one,
        // but in practice, transcribed music online seems to
        // follow the rule that the stem's duration is determined
        // by the shortest contained duration.
        if (noteTime && noteTime < minStemTime) {
          duration = noteDuration;
          minStemTime = noteTime;
        }
        // After a duration, look for a tie mark.  Individual notes
        // within a stem can be tied.
        if (index < tokens.length && '-' == tokens[index]) {
          if (lastNote) {
            notes[notes.length - 1].tie = true;
          }
          index++;
        }
      }
      // The last thing in a chord should be a ].  If it isn't, then
      // this doesn't look like a stem after all, and return null.
      if (tokens[index] != ']') {
        return null;
      }
      index++;
    } else if (index < tokens.length && /[A-Ga-g]/.test(tokens[index])) {
      // Grab a single note.
      lastNote = {
        pitch: applyAccent(tokens[index++], key, accent),
        tie: false,
        duration: '',
        time: 1
      }
      lastNote.frequency = pitchToFrequency(lastNote.pitch);
      notes.push(lastNote);
    } else if (index < tokens.length && /^[xzXZ]$/.test(tokens[index])) {
      // Grab a rest - no pitch.
      index++;
    } else {
      // Something we don't recognize - not a stem.
      return null;
    }
    // Right after a [chord], note, or rest, look for a duration marking.
    if (index < tokens.length && /^(?![\s%!]).*[\d\/]/.test(tokens[index])) {
      duration = tokens[index++];
      noteTime = durationToTime(duration);
      // Apply the duration to all the ntoes in the stem.
      // NOTE: spec suggests multiplying this duration, but that
      // idiom is not seen (so far) in practice.
      for (j = 0; j < notes.length; ++j) {
        notes[j].duration = duration;
        notes[j].time = noteTime;
      }
    }
    // Then look for a trailing tie marking.  Will tie every note in a chord.
    if (index < tokens.length && '-' == tokens[index]) {
      index++;
      for (j = 0; j < notes.length; ++j) {
        notes[j].tie = true;
      }
    }
    if (accent.dynamics) {
      velocity = accent.dynamics;
      for (j = 0; j < notes.length; ++j) {
        notes[j].velocity = velocity;
      }
    }
    return {
      index: index,
      stem: {
        notes: notes,
        duration: duration,
        staccato: staccato,
        time: durationToTime(duration)
      }
    };
  }
  // Normalizes pitch markings by stripping leading = if present.
  function stripNatural(pitch) {
    if (pitch.length > 0 && pitch.charAt(0) == '=') {
      return pitch.substr(1);
    }
    return pitch;
  }
  // Processes an accented pitch, automatically applying accidentals
  // that have accumulated within the measure, and also saving
  // explicit accidentals to continue to apply in the measure.
  function applyAccent(pitch, key, accent) {
    var m = /^(\^+|_+|=|)([A-Ga-g])(.*)$/.exec(pitch), letter;
    if (!m) { return pitch; }
    // Note that an accidental in one octave applies in other octaves.
    letter = m[2].toUpperCase();
    if (m[1].length > 0) {
      // When there is an explicit accidental, then remember it for
      // the rest of the measure.
      accent[letter] = m[1];
      return stripNatural(pitch);
    }
    if (accent.hasOwnProperty(letter)) {
      // Accidentals from this measure apply to unaccented notes.
      return stripNatural(accent[letter] + m[2] + m[3]);
    }
    if (key.hasOwnProperty(letter)) {
      // Key signatures apply by default.
      return stripNatural(key[letter] + m[2] + m[3]);
    }
    return stripNatural(pitch);
  }
  // Converts an ABC duration to a number (e.g., "/3"->0.333 or "11/2"->1.5).
  function durationToTime(duration) {
    var m = /^(\d*)(?:\/(\d*))?$|^(\/+)$/.exec(duration), n, d, i = 0, ilen;
    if (!m) return;
    if (m[3]) return Math.pow(0.5, m[3].length);
    d = (m[2] ? parseFloat(m[2]) : /\//.test(duration) ? 2 : 1);
    // Handle mixed frations:
    ilen = 0;
    n = (m[1] ? parseFloat(m[1]) : 1);
    if (m[2]) {
      while (ilen + 1 < m[1].length && n > d) {
        ilen += 1
        i = parseFloat(m[1].substring(0, ilen))
        n = parseFloat(m[1].substring(ilen))
      }
    }
    return i + (n / d);
  }
}

// wavetable is a table of names for nonstandard waveforms.
// The table maps names to objects that have wave: and freq:
// properties. The wave: property is a PeriodicWave to use
// for the oscillator.  The freq: property, if present,
// is a map from higher frequencies to more PeriodicWave
// objects; when a frequency higher than the given threshold
// is requested, the alternate PeriodicWave is used.
function makeWavetable(ac) {
  return (function(wavedata) {
    function makePeriodicWave(data) {
      var n = data.real.length,
          real = new Float32Array(n),
          imag = new Float32Array(n),
          j;
      for (j = 0; j < n; ++j) {
        real[j] = data.real[j];
        imag[j] = data.imag[j];
      }
      try {
        // Latest API naming.
        return ac.createPeriodicWave(real, imag);
      } catch (e) { }
      try {
        // Earlier API naming.
        return ac.createWaveTable(real, imag);
      } catch (e) { }
      return null;
    }
    function makeMultiple(data, mult, amt) {
      var result = { real: [], imag: [] }, j, n = data.real.length, m;
      for (j = 0; j < n; ++j) {
        m = Math.log(mult[Math.min(j, mult.length - 1)]);
        result.real.push(data.real[j] * Math.exp(amt * m));
        result.imag.push(data.imag[j] * Math.exp(amt * m));
      }
      return result;
    }
    var result = {}, k, d, n, j, ff, record, wave, pw;
    for (k in wavedata) {
      d = wavedata[k];
      wave = makePeriodicWave(d);
      if (!wave) { continue; }
      record = { wave: wave };
      // A strategy for computing higher frequency waveforms: apply
      // multipliers to each harmonic according to d.mult.  These
      // multipliers can be interpolated and applied at any number
      // of transition frequencies.
      if (d.mult) {
        ff = wavedata[k].freq;
        record.freq = {};
        for (j = 0; j < ff.length; ++j) {
          wave =
            makePeriodicWave(makeMultiple(d, d.mult, (j + 1) / ff.length));
          if (wave) { record.freq[ff[j]] = wave; }
        }
      }
      // This wave has some default filter settings.
      if (d.defs) {
        record.defs = d.defs;
      }
      result[k] = record;
    }
    return result;
  })({
    // Currently the only nonstandard waveform is "piano".
    // It is based on the first 32 harmonics from the example:
    // https://github.com/GoogleChrome/web-audio-samples
    // /blob/gh-pages/samples/audio/wave-tables/Piano
    // That is a terrific sound for the lowest piano tones.
    // For higher tones, interpolate to a customzed wave
    // shape created by hand, and apply a lowpass filter.
    piano: {
      real: [0, 0, -0.203569, 0.5, -0.401676, 0.137128, -0.104117, 0.115965,
             -0.004413, 0.067884, -0.00888, 0.0793, -0.038756, 0.011882,
             -0.030883, 0.027608, -0.013429, 0.00393, -0.014029, 0.00972,
             -0.007653, 0.007866, -0.032029, 0.046127, -0.024155, 0.023095,
             -0.005522, 0.004511, -0.003593, 0.011248, -0.004919, 0.008505],
      imag: [0, 0.147621, 0, 0.000007, -0.00001, 0.000005, -0.000006, 0.000009,
             0, 0.000008, -0.000001, 0.000014, -0.000008, 0.000003,
             -0.000009, 0.000009, -0.000005, 0.000002, -0.000007, 0.000005,
             -0.000005, 0.000005, -0.000023, 0.000037, -0.000021, 0.000022,
             -0.000006, 0.000005, -0.000004, 0.000014, -0.000007, 0.000012],
      // How to adjust the harmonics for the higest notes.
      mult: [1, 1, 0.18, 0.016, 0.01, 0.01, 0.01, 0.004,
                0.014, 0.02, 0.014, 0.004, 0.002, 0.00001],
      // The frequencies at which to interpolate the harmonics.
      freq: [65, 80, 100, 135, 180, 240, 620, 1360],
      // The default filter settings to use for the piano wave.
      // TODO: this approach attenuates low notes too much -
      // this should be fixed.
      defs: { wave: 'piano', gain: 0.5,
              attack: 0.002, decay: 0.25, sustain: 0.03, release: 0.1,
              decayfollow: 0.7,
              cutoff: 800, cutfollow: 0.1, resonance: 1, detune: 0.9994 }
    }
  });
}

// End of musical.js copy.


//////////////////////////////////////////////////////////////////////////
// SYNC, REMOVE SUPPORT
// sync() function
//////////////////////////////////////////////////////////////////////////

function gatherelts(args) {
  var elts = [], j, argcount = args.length, completion;
  // The optional last argument is a callback when the sync is triggered.
  if (argcount && $.isFunction(args[argcount - 1])) {
    completion = args[--argcount];
  }
  // Gather elements passed as arguments.
  for (j = 0; j < argcount; ++j) {
    if (!args[j]) {
      continue;  // Skip null args.
    } else if (args[j].constructor === $) {
      elts.push.apply(elts, args[j].toArray());  // Unpack jQuery.
    } else if ($.isArray(args[j])) {
      elts.push.apply(elts, args[j]);  // Accept an array.
    } else {
      elts.push(args[j]);  // Individual elements.
    }
  }
  return {
    elts: $.unique(elts),  // Remove duplicates.
    completion: completion
  };
}

function sync() {
  var a = gatherelts(arguments),
      elts = a.elts, completion = a.completion, j, ready = [];
  function proceed() {
    var cb = ready, j;
    ready = null;
    // Call completion before unblocking animation.
    if (completion) { completion(); }
    // Unblock all animation queues.
    for (j = 0; j < cb.length; ++j) { cb[j](); }
  }
  if (elts.length > 1) for (j = 0; j < elts.length; ++j) {
    queueWaitIfLoadingImg(elts[j]);
    $(elts[j]).queue(function(next) {
      if (ready) {
        ready.push(next);
        if (ready.length == elts.length) {
          proceed();
        }
      }
    });
  }
}

function remove() {
  var a = gatherelts(arguments),
      elts = a.elts, completion = a.completion, j, count = elts.length;
  for (j = 0; j < elts.length; ++j) {
    $(elts[j]).queue(function(next) {
      $(this).remove();
      count -= 1;
      if (completion && count == 0) { completion(); }
      next();
    });
  }
}

//////////////////////////////////////////////////////////////////////////
// JQUERY REGISTRATION
// Register all our hooks.
//////////////////////////////////////////////////////////////////////////

$.extend(true, $, {
  cssHooks: {
    turtlePenStyle: makePenStyleHook(),
    turtlePenDown: makePenDownHook(),
    turtleSpeed: makeTurtleSpeedHook(),
    turtleEasing: makeTurtleEasingHook(),
    turtleForward: makeTurtleForwardHook(),
    turtleTurningRadius: makeTurningRadiusHook(),
    turtlePosition: makeTurtleXYHook('turtlePosition', 'tx', 'ty', true),
    turtlePositionX: makeTurtleHook('tx', parseFloat, 'px', true),
    turtlePositionY: makeTurtleHook('ty', parseFloat, 'px', true),
    turtleRotation: makeTurtleHook('rot', maybeArcRotation, 'deg', true),
    turtleScale: makeTurtleXYHook('turtleScale', 'sx', 'sy', false),
    turtleScaleX: makeTurtleHook('sx', identity, '', false),
    turtleScaleY: makeTurtleHook('sy', identity, '', false),
    turtleTwist: makeTurtleHook('twi', normalizeRotation, 'deg', false),
    turtleHull: makeHullHook(),
    turtleTimbre: makeTimbreHook(),
    turtleVolume: makeVolumeHook()
  },
  cssNumber: {
    turtleRotation: true,
    turtleSpeed: true,
    turtleScale: true,
    turtleScaleX: true,
    turtleScaleY: true,
    turtleTwist: true
  },
  support: {
    turtle: true
  }
});
$.extend(true, $.fx, {
  step: {
    turtlePosition: makePairStep('turtlePosition', true),
    turtleRotation: makeRotationStep('turtleRotation'),
    turtleScale: makePairStep('turtleScale', false),
    turtleTwist: makeRotationStep('turtleTwist')
  },
  speeds: {
    turtle: 0
  }
});

//////////////////////////////////////////////////////////////////////////
// FUNCTION WRAPPERS
// Wrappers for all API functions
//////////////////////////////////////////////////////////////////////////

function helpwrite(text) {
  see.html('<aside style="line-height:133%;word-break:normal;' +
           'white-space:normal">' + text + '</aside>');
}
function globalhelp(obj) {
  var helptable = $.extend({}, dollar_turtle_methods, turtlefn, extrahelp),
      helplist, j;
  if (obj && (!$.isArray(obj.helptext))) {
    if (obj in helptable) {
      obj = helptable[obj];
    }
  }
  if (obj && $.isArray(obj.helptext) && obj.helptext.length) {
    for (j = 0; j < obj.helptext.length; ++j) {
      var text = obj.helptext[j];
      helpwrite(text.replace(/<(u)>/g,
          '<$1 style="border:1px solid black;text-decoration:none;' +
          'word-break:keep-all;white-space:nowrap">').replace(/<(mark)>/g,
          '<$1 style="border:1px solid blue;color:blue;text-decoration:none;' +
          'word-break:keep-all;white-space:nowrap;cursor:pointer;" ' +
          'onclick="see.enter($(this).text())">'));
    }
    return helpok;
  }
  if (typeof obj == 'number') {
    helpwrite('Equal to the number ' + obj + '.');
    return helpok;
  }
  if (typeof obj == 'boolean') {
    helpwrite('Equal to the boolean value ' + obj + '.');
    return helpok;
  }
  if (obj === null) {
    helpwrite('The special null value represents the absence of a value.');
    return helpok;
  }
  if (obj === undefined) {
    helpwrite('This is an unassigned value.');
    return helpok;
  }
  if (obj === global) {
    helpwrite('The global window object represents the browser window.');
    return helpok;
  }
  if (obj === document) {
    helpwrite('The HTML document running the program.');
    return helpok;
  }
  if (obj === jQuery) {
    helpwrite('The jQuery function.  Read about it at ' +
        '<a href="http://learn.jquery.com/" target="_blank">jquery.com</a>.');
    return helpok;
  }
  if (obj && obj != globalhelp) {
    helpwrite('No help available for ' + obj);
    return helpok;
  }
  helplist = [];
  for (var name in helptable) {
    if (helptable[name].helptext && helptable[name].helptext.length &&
        (!(name in global) || typeof(global[name]) == 'function')) {
      helplist.push(name);
    }
  }
  helplist.sort(function(a, b) {
    if (a.length != b.length) { return a.length - b.length; }
    if (a < b) { return -1; }
    if (a > b) { return 1; }
    return 0;
  });
  helpwrite("help available for: " + helplist.map(function(x) {
     return '<mark style="border:1px solid blue;color:blue;text-decoration:none;' +
       'word-break:keep-all;white-space:nowrap;cursor:pointer;" ' +
       'onclick="see.enter($(this).text())">' + x + '</mark>';
  }).join(" "));
  return helpok;
}
globalhelp.helptext = [];

function canMoveInstantly(sel) {
  var atime, elem;
  // True if the selector names a single element with no animtation
  // queue and currently moving at speed Infinity.
  return sel.length == 1 && canElementMoveInstantly(sel[0]) && sel[0];
}

function canElementMoveInstantly(elem) {
  // True if the element has no animtation queue and currently is
  // moving at speed Infinity.
  var atime;
  return (elem && $.queue(elem).length == 0 &&
      (!elem.parentElement || !elem.parentElement.style.transform) &&
      ((atime = animTime(elem)) === 0 || $.fx.speeds[atime] === 0));
}

function visiblePause(elem, seconds) {
  var ms;
  if (seconds == null) {
    if (canElementMoveInstantly(elem)) {
      return;
    }
    ms = animTime(elem);
  } else {
    ms = seconds * 1000;
  }
  var thissel = $(elem);
  if (ms > 0) {
    thissel.delay(ms);
  }
}

function doNothing() {}

// When using continuation-passing-style (or await-defer), the
// common design pattern is for the last argument of a function
// to be a "continuation" function that is invoked exactly once when
// the aync action requested by the function is completed.  For example,
// the last argument of "lt 90, fn" is a function that is called when
// the turtle has finished animating left by 90 degrees.
// This function returns that last argument if it is a function and
// if the argument list is longer than argcount, or null otherwise.
function continuationArg(args, argcount) {
  argcount = argcount || 0;
  if (args.length <= argcount) { return null; }
  var lastarg = args[args.length - 1];
  if (typeof(lastarg) === 'function' && !lastarg.helpname) { return lastarg; }
  return null;
}

// This function helps implement the continuation-passing-style
// design pattern for turtle animation functions.  It examines the "this"
// jQuery object and the argument list.  If a continuation callback
// function is present, then it returns an object that provides:
//    name: the name of the function.
//    args: the argument list without the callback function.
//    appear: a callback function to be called this.length times,
//        as each of the elements' animations begins.  When the jth
//        element is animating, appear(j) should be called.
//    resolve: a callback function to be called this.length times,
//        as each of the elements' animations completes; when the jth
//        element completes, resolve(j) should be called.  The last time
//        it is called, it will trigger the continuation callback, if any.
//        Call resolve(j, true) if a corner pen state should be marked.
//    resolver: a function that returns a closure that calls resolve(j).
//    start: a function to be called once to enable triggering of the callback.
// the last argument in an argument list if it is a function, and if the
// argument list is longer than "argcount" in length.
function setupContinuation(thissel, name, args, argcount) {
  var done = continuationArg(args, argcount),
      mainargs = !done ? args :
          Array.prototype.slice.call(args, 0, args.length - 1),
      length = thissel ? thissel.length || 0 : 0,
      countdown = length + 1,
      sync = true,
      debugId = debug.nextId();
  function resolve(j, corner) {
    if (j != null) {
      var elem = thissel && thissel[j];
      if (corner && elem) {
        flushPenState(elem, $.data(elem, 'turtleData'), true);
      }
      debug.reportEvent('resolve', [name, debugId, length, j, elem]);
    }
    if ((--countdown) == 0) {
      // A subtlety: if we still have not yet finished setting things up
      // when the callback is triggered, it means that we are synchronous
      // to the original call.  For execution-order consistency, we never
      // want to trigger the users' callback synchronously. So we use a
      // timeout in this case.
      if (done) {
        if (sync) {
          async_pending += 1;
          setTimeout(function() {
            async_pending -= 1;
            done();
          }, 0);
        } else {
          done();
        }
      }
    }
  }
  function appear(j) {
    if (j != null) {
      debug.reportEvent('appear',
          [name, debugId, length, j, thissel && thissel[j], mainargs]);
    }
  }
  debug.reportEvent('enter', [name, debugId, length, mainargs]);
  return {
    name: name,
    args: mainargs,
    appear: appear,
    resolve: resolve,
    resolver: function(j, c) { return function() { resolve(j, c); }; },
    exit: function exit() {
      debug.reportEvent('exit', [name, debugId, length, mainargs]);
      // Resolve one extra countdown; this is needed for a done callback
      // in the case where thissel.length is zero.
      resolve(null);
      sync = false;
    }
  };
}

// Wrapcommand does boilerplate setup for turtle motion commands like "fd",
// providing uniform async support.  Commands wrapped by wrapcommand should
// apply an action once for each element of the "this" jQuery selector,
// calling "cc.appear(j)" as the action for the jth element begins and
// "cc.resolve(j)" as it ends.  The wrapper uses setupContinuation to deal
// with an optional last continuation argument and notification of the
// debugger.  All arguments other than the continuation callback are passed
// through to the underying function, adding "cc" as the first argument.
function wrapcommand(name, reqargs, helptext, fn) {
  var wrapper = function commandwrapper() {
    checkForHungLoop(name);
    if (interrupted) { throw new Error(name + ' interrupted'); }
    var cc = setupContinuation(this, name, arguments, reqargs),
        args = [cc].concat($.makeArray(cc.args)),
        result;
    try {
      result = fn.apply(this, args);
    } finally {
      cc.exit();
    }
    return result;
  }
  return wrapraw(name, helptext, wrapper);
}

// Wrappredicate does boilerplate setup for turtle predicates like "touches".
// When these predicates are used on a turtle that has many queued animations,
// a warning message is printed suggesting that "speed Infinity" or
// "await done defer()" should be considered.
function wrappredicate(name, helptext, fn) {
  var wrapper = function predicatewrapper() {
    checkForHungLoop(name);
    if (interrupted) { throw new Error(name + ' interrupted'); }
    checkPredicate(name, this);
    return fn.apply(this, arguments);
  };
  return wrapraw(name, helptext, wrapper);
}

// Wrapglobalcommand does boilerplate setup for global commands that should
// queue on the main turtle queue when there is a main turtle, but that
// should execute immediately otherwise.
function wrapglobalcommand(name, helptext, fn, fnfilter) {
  var wrapper = function globalcommandwrapper() {
    checkForHungLoop(name);
    if (interrupted) { throw new Error(name + ' interrupted'); }
    var early = null;
    var argcount = 0;
    var animate = global_turtle;
    if (fnfilter) {
      early = fnfilter.apply(null, arguments);
      argcount = arguments.length;
      animate = global_turtle_animating();
    }
    if (animate) {
      var thissel = $(global_turtle).eq(0),
          args = arguments,
          cc = setupContinuation(thissel, name, arguments, argcount);
      thissel.plan(function(j, elem) {
        cc.appear(j);
        fn.apply(early, args);
        this.plan(cc.resolver(j));
      });
      cc.exit();
    } else {
      cc = setupContinuation(null, name, arguments, argcount);
      fn.apply(early, arguments);
      cc.exit();
    }
    if (early) {
      if (early.result && early.result.constructor === jQuery && global_turtle) {
        sync(global_turtle, early.result);
      }
      return early.result;
    }
  };
  return wrapraw(name, helptext, wrapper);
}

function wrapwindowevent(name, helptext) {
  return wrapraw(name, helptext, function(d, fn) {
    var forKey = /^key/.test(name),
        forMouse = /^mouse|click$/.test(name),
        filter = forMouse ? 'input,button' : forKey ?
            'textarea,input:not([type]),input[type=text],input[type=password]'
            : null;
    if (forKey) { focusWindowIfFirst(); }
    if (fn == null && typeof(d) == 'function') { fn = d; d = null; }
    $(global).on(name + '.turtleevent', null, d, !filter ? fn : function(e) {
      if (interrupted) return;
      if ($(e.target).closest(filter).length) { return; }
      return fn.apply(this, arguments);
    });
  });
}

function windowhasturtleevent() {
  var events = $._data(global, 'events');
  if (!events) return false;
  for (var type in events) {
    var entries = events[type];
    for (var j = 0; j < entries.length; ++j) {
      if (entries[j].namespace == 'turtleevent') return true;
    }
  }
  return false;
}

// Wrapraw sets up help text for a function (such as "sqrt") that does
// not need any other setup.
function wrapraw(name, helptext, fn) {
  fn.helpname = name;
  fn.helptext = helptext;
  return fn;
}

//////////////////////////////////////////////////////////////////////////
// BASIC TURTLE MOTIONS
// Generic functions to handle symmetric pairs of motions.
//////////////////////////////////////////////////////////////////////////

// Deals with both rt and lt by negating degrees if cc.name is "lt".
function rtlt(cc, degrees, radius) {
  if (degrees == null) {
    degrees = 90;  // zero-argument default.
  } else {
    degrees = normalizeRotationDelta(degrees);
  }
  var elem, left = (cc.name === 'lt'), intick = insidetick;
  if ((elem = canMoveInstantly(this)) &&
      (radius === 0 || (radius == null && getTurningRadius(elem) === 0))) {
    cc.appear(0);
    doQuickRotate(elem, (left ? -degrees : degrees));
    cc.resolve(0);
    return this;
  }
  var operator = (left ? "-=" : "+=");
  if (radius == null) {
    this.plan(function(j, elem) {
      cc.appear(j);
      this.animate({turtleRotation: operator + cssNum(degrees || 0) + 'deg'},
          animTime(elem, intick), animEasing(elem), cc.resolver(j));
    });
    return this;
  } else {
    this.plan(function(j, elem) {
      cc.appear(j);
      var state = getTurtleData(elem),
          oldRadius = state.turningRadius,
          newRadius = (degrees < 0) ? -radius : radius,
          addCorner = null;
      if (state.style && state.down) {
        addCorner = (function() {
          var oldPos = getCenterInPageCoordinates(elem),
              oldTs = readTurtleTransform(elem, true),
              oldTransform = totalTransform2x2(elem.parentElement);
          return (function() {
            addArcBezierPaths(
              state.corners[0],
              oldPos,
              oldTs.rot,
              oldTs.rot + (left ? -degrees : degrees),
              newRadius * (state.oldscale ? oldTs.sy : 1),
              oldTransform);
          });
        })();
      }
      state.turningRadius = newRadius;
      this.animate({turtleRotation: operator + cssNum(degrees) + 'deg'},
          animTime(elem, intick), animEasing(elem));
      this.plan(function() {
        if (addCorner) addCorner();
        state.turningRadius = oldRadius;
        cc.resolve(j, true);
      });
    });
    return this;
  }
}

// Deals with both fd and bk by negating amount if cc.name is 'bk'.
function fdbk(cc, amount) {
  if (amount == null) {
    amount = 100;  // zero-argument default.
  }
  if (cc.name === 'bk') {
    amount = -amount;
  }
  var elem, intick = insidetick;
  if ((elem = canMoveInstantly(this))) {
    cc.appear(0);
    doQuickMove(elem, amount, 0);
    cc.resolve(0, true);
    return this;
  }
  this.plan(function(j, elem) {
    cc.appear(j);
    this.animate({turtleForward: '+=' + cssNum(amount || 0) + 'px'},
        animTime(elem, intick), animEasing(elem), cc.resolver(j, true));
  });
  return this;
}

//////////////////////////////////////////////////////////////////////////
// CARTESIAN MOVEMENT FUNCTIONS
//////////////////////////////////////////////////////////////////////////

function slide(cc, x, y) {
  if ($.isArray(x)) {
    y = x[1];
    x = x[0];
  }
  if (!y) { y = 0; }
  if (!x) { x = 0; }
  var intick = insidetick;
  this.plan(function(j, elem) {
    cc && cc.appear(j);
    this.animate({turtlePosition: displacedPosition(elem, y, x)},
        animTime(elem, intick), animEasing(elem), cc && cc.resolver(j, true));
  });
  return this;
}

function movexy(cc, x, y) {
  if ($.isArray(x)) {
    y = x[1];
    x = x[0];
  }
  if (!y) { y = 0; }
  if (!x) { x = 0; }
  var elem, intick = insidetick;
  if ((elem = canMoveInstantly(this))) {
    cc && cc.appear(0);
    doQuickMoveXY(elem, x, y);
    cc && cc.resolve(0);
    return this;
  }
  this.plan(function(j, elem) {
    cc && cc.appear(j);
    var tr = getElementTranslation(elem);
    this.animate(
      { turtlePosition: cssNum(tr[0] + x) + ' ' + cssNum(tr[1] - y) },
      animTime(elem, intick), animEasing(elem), cc && cc.resolver(j, true));
  });
  return this;
}

function moveto(cc, x, y) {
  var position = x, localx = 0, localy = 0, limit = null, intick = insidetick;
  if ($.isNumeric(position) && $.isNumeric(y)) {
    // moveto x, y: use local coordinates.
    localx = parseFloat(position);
    localy = parseFloat(y);
    position = null;
    limit = null;
  } else if ($.isArray(position)) {
    // moveto [x, y], limit: use local coordinates (limit optional).
    localx = position[0];
    localy = position[1];
    position = null;
    limit = y;
  } else if ($.isNumeric(y)) {
    // moveto obj, limit: limited motion in the direction of obj.
    limit = y;
  }
  // Otherwise moveto {pos}, limit: absolute motion with optional limit.
  this.plan(function(j, elem) {
    var pos = position;
    if (pos === null) {
      pos = $(homeContainer(elem)).pagexy();
    }
    if (pos && !isPageCoordinate(pos)) {
      try {
        pos = $(pos).pagexy();
      } catch (e) {
        return;
      }
    }
    if (!pos || !isPageCoordinate(pos)) return;
    if ($.isWindow(elem)) {
      cc && cc.appear(j);
      scrollWindowToDocumentPosition(pos, limit);
      cc && cc.resolve(j);
      return;
    } else if (elem.nodeType === 9) {
      return;
    }
    cc && cc.appear(j);
    this.animate({turtlePosition:
        computeTargetAsTurtlePosition(elem, pos, limit, localx, localy)},
        animTime(elem, intick), animEasing(elem), cc && cc.resolver(j, true));
  });
  return this;
}

// Deals with jump, jumpxy, and jumpto functions
function makejump(move) {
  return (function(cc, x, y) {
    this.plan(function(j, elem) {
      cc.appear(j);
      var down = this.css('turtlePenDown');
      this.css({turtlePenDown: 'up'});
      move.call(this, null, x, y);
      this.plan(function() {
        this.css({turtlePenDown: down});
        cc.resolve(j, true);
      });
    });
    return this;
  });
}

//////////////////////////////////////////////////////////////////////////
// SCALING FUNCTIONS
// Support for old-fashioned scaling and new.
//////////////////////////////////////////////////////////////////////////

function elemOldScale(elem) {
  var state = $.data(elem, 'turtleData');
  return state && (state.oldscale != null) ? state.oldscale : 1;
}

function scaleCmd(cc, valx, valy) {
  growImpl.call(this, true, cc, valx, valy);
}

function grow(cc, valx, valy) {
  growImpl.call(this, false, cc, valx, valy);
}

function growImpl(oldscale, cc, valx, valy) {
  if (valy === undefined) { valy = valx; }
  // Disallow scaling to zero using this method.
  if (!valx || !valy) { valx = valy = 1; }
  var intick = insidetick;
  this.plan(function(j, elem) {
    if (oldscale) {
      getTurtleData(elem).oldscale *= valy;
    }
    cc.appear(j);
    if ($.isWindow(elem) || elem.nodeType === 9) {
      cc.resolve(j);
      return;
    }
    var c = $.map($.css(elem, 'turtleScale').split(' '), parseFloat);
    if (c.length === 1) { c.push(c[0]); }
    c[0] *= valx;
    c[1] *= valy;
    this.animate({turtleScale: $.map(c, cssNum).join(' ')},
          animTime(elem, intick), animEasing(elem), cc.resolver(j));
  });
  return this;
}

//////////////////////////////////////////////////////////////////////////
// DOT AND BOX FUNCTIONS
// Support for animated drawing of dots and boxes.
//////////////////////////////////////////////////////////////////////////

function drawingScale(elem, oldscale) {
  var totalParentTransform = totalTransform2x2(elem.parentElement),
      simple = isone2x2(totalParentTransform),
      scale = simple ? 1 : decomposeSVD(totalParentTransform)[1];
  return scale * elemOldScale(elem);
}

function animatedDotCommand(fillShape) {
  var intick = insidetick;
  return (function(cc, style, diameter) {
    if ($.isNumeric(style)) {
      // Allow for parameters in either order.
      var t = style;
      style = diameter;
      diameter = t;
    }
    if (diameter == null) { diameter = 8.8; }
    this.plan(function(j, elem) {
      var state = getTurtleData(elem),
          penStyle = state.style;
      if (!style) {
        // If no color is specified, default to pen color, or black if no pen.
        style = (penStyle && (penStyle.fillStyle || penStyle.strokeStyle)) ||
            'black';
      }
      cc.appear(j);
      var c = this.pagexy(),
          ts = readTurtleTransform(elem, true),
          ps = parsePenStyle(style, 'fillStyle'),
          drawOnCanvas = getDrawOnCanvas(state),
          sx = drawingScale(elem),
          targetDiam = diameter * sx,
          animDiam = Math.max(0, targetDiam - 2),
          finalDiam = targetDiam + (ps.eraseMode ? 2 : 0),
          hasAlpha = /rgba|hsla/.test(ps.fillStyle);
      if (null == ps.lineWidth && penStyle && penStyle.lineWidth) {
        ps.lineWidth = penStyle.lineWidth;
      }
      if (canMoveInstantly(this)) {
        fillShape(drawOnCanvas, c, finalDiam, ts.rot, ps, true);
        cc.resolve(j);
      } else {
        this.queue(function(next) {
          $({radius: 0}).animate({radius: animDiam}, {
            duration: animTime(elem, intick),
            step: function() {
              if (!hasAlpha) {
                fillShape(drawOnCanvas, c, this.radius, ts.rot, ps, false);
              }
            },
            complete: function() {
              fillShape(drawOnCanvas, c, finalDiam, ts.rot, ps, true);
              cc.resolve(j);
              next();
            }
          })
        });
      }
    });
    return this;
  });
}

function fillDot(drawOnCanvas, position, diameter, rot, style) {
  var ctx = drawOnCanvas.getContext('2d');
  ctx.save();
  applyPenStyle(ctx, style);
  if (diameter === Infinity) {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillRect(0, 0, drawOnCanvas.width, drawOnCanvas.height);
  } else {
    setCanvasPageTransform(ctx, drawOnCanvas);
    ctx.beginPath();
    ctx.arc(position.pageX, position.pageY, diameter / 2, 0, 2*Math.PI, false);
    ctx.closePath();
    ctx.fill();
    if (style.strokeStyle) {
      ctx.stroke();
    }
  }
  ctx.restore();
}

function fillBox(drawOnCanvas, position, diameter, rot, style) {
  var ctx = drawOnCanvas.getContext('2d');
  ctx.save();
  applyPenStyle(ctx, style);
  if (diameter === Infinity) {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillRect(0, 0, drawOnCanvas.width, drawOnCanvas.height);
  } else {
    var s = Math.sin((rot + 45) / 180 * Math.PI),
        c = Math.cos((rot + 45) / 180 * Math.PI),
        hdx = diameter * c / Math.SQRT2,
        hdy = diameter * s / Math.SQRT2;
    setCanvasPageTransform(ctx, drawOnCanvas);
    ctx.beginPath();
    ctx.moveTo(position.pageX - hdx, position.pageY - hdy);
    ctx.lineTo(position.pageX - hdy, position.pageY + hdx);
    ctx.lineTo(position.pageX + hdx, position.pageY + hdy);
    ctx.lineTo(position.pageX + hdy, position.pageY - hdx);
    ctx.closePath();
    ctx.fill();
    if (style.strokeStyle) {
      ctx.stroke();
    }
  }
  ctx.restore();
}

function fillArrow(drawOnCanvas, position, diameter, rot, style, drawhead) {
  var ctx = drawOnCanvas.getContext('2d');
  ctx.save();
  applyPenStyle(ctx, style);
  if (!style.strokeStyle && style.fillStyle) {
    ctx.strokeStyle = style.fillStyle;
  }
  if (diameter !== Infinity) {
    var c = Math.sin(rot / 180 * Math.PI),
        s = -Math.cos(rot / 180 * Math.PI),
        w = style.lineWidth || 1.62,
        hx = position.pageX + diameter * c,
        hy = position.pageY + diameter * s,
        m = calcArrow(w, hx, hy, c, s),
        ds = diameter - m.hs,
        dx = ds * c,
        dy = ds * s;
    setCanvasPageTransform(ctx, drawOnCanvas);
    if (ds > 0) {
      ctx.beginPath();
      ctx.moveTo(position.pageX, position.pageY);
      ctx.lineTo(position.pageX + dx, position.pageY + dy);
      ctx.stroke();
    }
    if (drawhead) {
      drawArrowHead(ctx, m);
    }
  }
  ctx.restore();
}

//////////////////////////////////////////////////////////////////////////
// ARROW GEOMETRY
//////////////////////////////////////////////////////////////////////////
function calcArrow(w, x1, y1, cc, ss) {
  var hw = Math.max(w * 1.25, w + 2),
      hh = hw * 2,
      hs = hh - hw / 2;
  return {
      hs: hs,
      x1: x1,
      y1: y1,
      xm: x1 - cc * hs,
      ym: y1 - ss * hs,
      x2: x1 - ss * hw - cc * hh,
      y2: y1 + cc * hw - ss * hh,
      x3: x1 + ss * hw - cc * hh,
      y3: y1 - cc * hw - ss * hh
  };
}

function drawArrowHead(c, m) {
  c.beginPath();
  c.moveTo(m.x2, m.y2);
  c.lineTo(m.x1, m.y1);
  c.lineTo(m.x3, m.y3);
  c.quadraticCurveTo(m.xm, m.ym, m.x2, m.y2);
  c.closePath();
  c.fill();
}

function drawArrowLine(c, w, x0, y0, x1, y1) {
  var dx = x1 - x0,
      dy = y1 - y0,
      dd = Math.sqrt(dx * dx + dy * dy),
      cc = dx / dd,
      ss = dy / dd;
  var m = calcArrow(w, x1, y1, cc, ss);
  if (dd > m.hs) {
    c.beginPath();
    c.moveTo(x0, y0);
    c.lineTo(m.xm, m.ym);
    c.lineWidth = w;
    c.stroke();
  }
  drawArrowHead(c, m);
}

//////////////////////////////////////////////////////////////////////////
// VOICE SYNTHESIS
// Method for uttering words.
//////////////////////////////////////////////////////////////////////////
function utterSpeech(words, cb) {
  var pollTimer = null;
  function complete() {
    if (pollTimer) { clearInterval(pollTimer); }
    if (cb) { cb(); }
  }
  if (!global.speechSynthesis) {
    console.log('No speech synthesis: ' + words);
    complete();
    return;
  }
  try {
    var msg = new global.SpeechSynthesisUtterance(words);
    msg.addEventListener('end', complete);
    msg.addEventListener('error', complete);
    msg.lang = navigator.language || 'en-GB';
    global.speechSynthesis.speak(msg);
    pollTimer = setInterval(function() {
      // Chrome speech synthesis fails to deliver an 'end' event
      // sometimes, so we also poll every 250ms.
      if (global.speechSynthesis.pending || global.speechSynthesis.speaking) return;
      complete();
    }, 250);
  } catch (e) {
    if (global.console) { global.console.log(e); }
    complete();
  }
}

//////////////////////////////////////////////////////////////////////////
// TURTLE FUNCTIONS
// Turtle methods to be registered as jquery instance methods.
//////////////////////////////////////////////////////////////////////////

var turtlefn = {
  rt: wrapcommand('rt', 1,
  ["<u>rt(degrees)</u> Right turn. Pivots clockwise by some degrees: " +
      "<mark>rt 90</mark>",
   "<u>rt(degrees, radius)</u> Right arc. Pivots with a turning radius: " +
      "<mark>rt 90, 50</mark>"], rtlt),
  lt: wrapcommand('lt', 1,
  ["<u>lt(degrees)</u> Left turn. Pivots counterclockwise by some degrees: " +
      "<mark>lt 90</mark>",
   "<u>lt(degrees, radius)</u> Left arc. Pivots with a turning radius: " +
      "<mark>lt 90, 50</mark>"], rtlt),
  fd: wrapcommand('fd', 1,
  ["<u>fd(pixels)</u> Forward. Moves ahead by some pixels: " +
      "<mark>fd 100</mark>"], fdbk),
  bk: wrapcommand('bk', 1,
  ["<u>bk(pixels)</u> Back. Moves in reverse by some pixels: " +
      "<mark>bk 100</mark>"], fdbk),
  slide: wrapcommand('slide', 1,
  ["<u>move(x, y)</u> Slides right x and forward y pixels without turning: " +
      "<mark>slide 50, 100</mark>"], slide),
  movexy: wrapcommand('movexy', 1,
  ["<u>movexy(x, y)</u> Changes graphing coordinates by x and y: " +
      "<mark>movexy 50, 100</mark>"], movexy),
  moveto: wrapcommand('moveto', 1,
  ["<u>moveto(x, y)</u> Move to graphing coordinates (see <u>getxy</u>): " +
      "<mark>moveto 50, 100</mark>",
   "<u>moveto(obj)</u> Move to page coordinates " +
      "or an object on the page (see <u>pagexy</u>): " +
      "<mark>moveto lastmousemove</mark>"], moveto),
  jump: wrapcommand('jump', 1,
  ["<u>jump(x, y)</u> Move without drawing (compare to <u>slide</u>): " +
      "<mark>jump 0, 50</mark>"], makejump(slide)),
  jumpxy: wrapcommand('jumpxy', 1,
  ["<u>jumpxy(x, y)</u> Move without drawing (compare to <u>movexy</u>): " +
      "<mark>jump 0, 50</mark>"], makejump(movexy)),
  jumpto: wrapcommand('jumpto', 1,
  ["<u>jumpto(x, y)</u> Move without drawing (compare to <u>moveto</u>): " +
      "<mark>jumpto 50, 100</mark>"], makejump(moveto)),
  turnto: wrapcommand('turnto', 1,
  ["<u>turnto(degrees)</u> Turn to a direction. " +
      "North is 0, East is 90: <mark>turnto 270</turnto>",
   "<u>turnto(x, y)</u> Turn to graphing coordinates: " +
      "<mark>turnto 50, 100</mark>",
   "<u>turnto(obj)</u> Turn to page coordinates or an object on the page: " +
      "<mark>turnto lastmousemove</mark>"],
  function turnto(cc, bearing, y) {
    if ($.isNumeric(y) && $.isNumeric(bearing)) {
      // turnto x, y: convert to turnto [x, y].
      bearing = [bearing, y];
      y = null;
    }
    var intick = insidetick;
    this.plan(function(j, elem) {
      cc.appear(j);
      if ($.isWindow(elem) || elem.nodeType === 9) {
        cc.resolve(j);
        return;
      }
      // turnto bearing: just use the given absolute.
      var limit = null, ts, r, centerpos,
          targetpos = null, nlocalxy = null;
      if ($.isNumeric(bearing)) {
        r = convertToRadians(bearing);
        centerpos = getCenterInPageCoordinates(elem);
        targetpos = {
          pageX: centerpos.pageX + Math.sin(r) * 1024,
          pageY: centerpos.pageY - Math.cos(r) * 1024
        };
        limit = y;
      } else if ($.isArray(bearing)) {
        nlocalxy = computePositionAsLocalOffset(elem);
        nlocalxy[0] -= bearing[0];
        nlocalxy[1] -= bearing[1];
      } else if (isPageCoordinate(bearing)) {
        targetpos = bearing;
      } else {
        try {
          targetpos = $(bearing).pagexy();
        } catch(e) {
          cc.resolve(j);
          return;
        }
      }
      if (!nlocalxy) {
        nlocalxy = computePositionAsLocalOffset(elem, targetpos);
      }
      var dir = radiansToDegrees(Math.atan2(-nlocalxy[0], -nlocalxy[1]));
      ts = readTurtleTransform(elem, true);
      if (!(limit === null)) {
        r = convertToRadians(ts.rot);
        dir = limitRotation(ts.rot, dir, limit === null ? 360 : limit);
      }
      dir = ts.rot + normalizeRotation(dir - ts.rot);
      var oldRadius = this.css('turtleTurningRadius');
      this.css({turtleTurningRadius: 0});
      this.animate({turtleRotation: dir},
          animTime(elem, intick), animEasing(elem));
      this.plan(function() {
        this.css({turtleTurningRadius: oldRadius});
        cc.resolve(j);
      });
    });
    return this;
  }),
  home: wrapcommand('home', 0,
  ["<u>home()</u> Goes home. " +
      "Jumps to the center without drawing: <mark>do home</mark>"],
  function home(cc, container) {
    this.plan(function(j, elem) {
      cc.appear(j);
      var down = this.css('turtlePenDown'),
          radius = this.css('turtleTurningRadius'),
          hc = container || homeContainer(elem);
      this.css({turtlePenDown: 'up', turtleTurningRadius: 0 });
      this.css({
        turtlePosition:
          computeTargetAsTurtlePosition(
              elem, $(hc).pagexy(), null, 0, 0),
        turtleRotation: 0,
        turtleScale: 1});
      this.css({turtlePenDown: down, turtleTurningRadius: radius });
      cc.resolve(j);
    });
    return this;
  }),
  copy: wrapcommand('copy', 0,
  ["<u>copy()</u> makes a new turtle that is a copy of this turtle."],
  function copy(cc) {
    var t2 =  this.clone().insertAfter(this);
    t2.hide();
    // t2.plan doesn't work here.
    this.plan(function(j, elem) {
      cc.appear(j);

      //copy over turtle data:
      var olddata = getTurtleData(this);
      var newdata = getTurtleData(t2);
      for (var k in olddata) { newdata[k] = olddata[k]; }

      // copy over style attributes:
      t2.attr('style', this.attr('style'));

      // copy each thing listed in css hooks:
      for (var property in $.cssHooks) {
        var value = this.css(property);
        t2.css(property, value);
      }

      // copy attributes, just in case:
      var attrs = this.prop("attributes");
      for (var i in attrs) {
        t2.attr(attrs[i].name, attrs[i].value);
      }

      // copy the canvas:
      var t2canvas = t2.canvas();
      var tcanvas = this.canvas();
      if (t2canvas && tcanvas) {
        t2canvas.width = tcanvas.width;
        t2canvas.height = tcanvas.height;
        var newCanvasContext = t2canvas.getContext('2d');
        newCanvasContext.drawImage(tcanvas, 0, 0)
      }

      t2.show();

      cc.resolve(j);
    }); // pass in our current clone, otherwise things apply to the wrong clone
    sync(t2, this);
    return t2;
  }),
  pen: wrapcommand('pen', 1,
  ["<u>pen(color, size)</u> Selects a pen. " +
      "Chooses a color and/or size for the pen: " +
      "<mark>pen red</mark>; <mark>pen 0</mark>; " +
      "<mark>pen erase</mark>; " +
      "<mark>pen blue, 5</mark>.",
   "<u>pen(on-or-off)</u> " +
      "Turns the pen on or off: " +
      "<mark>pen off</mark>; <mark>pen on</mark>."
  ],
  function pen(cc, penstyle, lineWidth) {
    var args = autoArgs(arguments, 1, {
      lineCap: /^(?:butt|square|round)$/,
      lineJoin: /^(?:bevel|round|miter)$/,
      lineWidth: $.isNumeric,
      penStyle: '*'
    });
    penstyle = args.penStyle;
    if (penstyle && (typeof(penstyle) == "function") && (
        penstyle.helpname || penstyle.name)) {
      // Deal with "tan" and "fill".
      penstyle = (penstyle.helpname || penstyle.name);
    }
    if (args.lineWidth === 0 || penstyle === null) {
      penstyle = "none";
    } else if (penstyle === undefined) {
      penstyle = 'black';
    } else if ($.isPlainObject(penstyle)) {
      penstyle = writePenStyle(penstyle);
    }
    var intick = insidetick;
    this.plan(function(j, elem) {
      cc.appear(j);
      var animate = !invisible(elem) && !canMoveInstantly(this),
          oldstyle = animate && parsePenStyle(this.css('turtlePenStyle')),
          olddown = oldstyle && ('down' == this.css('turtlePenDown')),
          moved = false;
      if (penstyle === false || penstyle === true ||
          penstyle == 'down' || penstyle == 'up') {
        this.css('turtlePenDown', penstyle);
        moved = true;
      } else {
        if (args.lineWidth) {
          penstyle += ";lineWidth:" + args.lineWidth;
        }
        if (args.lineCap) {
          penstyle += ";lineCap:" + args.lineCap;
        }
        if (args.lineJoin) {
          penstyle += ";lineJoin:" + args.lineJoin;
        }
        this.css('turtlePenStyle', penstyle);
        this.css('turtlePenDown', penstyle == 'none' ? 'up' : 'down');
      }
      if (animate) {
        // A visual indicator of a pen color change.
        var style = parsePenStyle(this.css('turtlePenStyle')),
            color = (style && (style.strokeStyle ||
                        (style.savePath && 'gray'))) ||
                    (oldstyle && oldstyle.strokeStyle) || 'gray',
            target = {},
            newdown = style && 'down' == this.css('turtlePenDown'),
            pencil = new Turtle(color + ' pencil', this.parent()),
            distance = this.height();
        pencil.css({
          zIndex: 1,
          turtlePosition: computeTargetAsTurtlePosition(
              pencil.get(0), this.pagexy(), null, 0, 0),
          turtleRotation: this.css('turtleRotation'),
          turtleSpeed: Infinity
        });
        if (!olddown) {
          pencil.css({ turtleForward: "+=" + distance, opacity: 0 });
          if (newdown) {
            target.turtleForward = "-=" + distance;
            target.opacity = 1;
          }
        } else {
          if (!newdown) {
            target.turtleForward = "+=" + distance;
            target.opacity = 0;
          }
        }
        if (oldstyle && !style && olddown == "down") {
          target.turtleForward = "+=" + distance;
          target.opacity = 0;
        } else if (oldstyle != style && (!oldstyle || !style ||
              oldstyle.strokeStyle != style.strokeStyle)) {
          pencil.css({ opacity: 0 });
          target.opacity = 1;
        }
        pencil.animate(target, animTime(elem, intick));
        this.queue(function(next) {
          pencil.done(function() {
            pencil.remove();
            next();
          });
        });
      }
      this.plan(function() {
        cc.resolve(j);
      });
    });
    return this;
  }),
  fill: wrapcommand('fill', 0,
  ["<u>fill(color)</u> Fills a path traced using " +
      "<u>pen path</u>: " +
      "<mark>pen path; rt 100, 90; fill blue</mark>"],
  function fill(cc, style) {
    if (!style) { style = 'none'; }
    else if ($.isPlainObject(style)) {
      style = writePenStyle(style);
    }
    var ps = parsePenStyle(style, 'fillStyle');
    this.plan(function(j, elem) {
      cc.appear(j);
      endAndFillPenPath(elem, ps);
      cc.resolve(j);
    });
    return this;
  }),
  dot: wrapcommand('dot', 0,
  ["<u>dot(color, diameter)</u> Draws a dot. " +
      "Color and diameter are optional: " +
      "<mark>dot blue</mark>"], animatedDotCommand(fillDot)),
  box: wrapcommand('box', 0,
  ["<u>box(color, size)</u> Draws a box. " +
      "Color and size are optional: " +
      "<mark>dot blue</mark>"], animatedDotCommand(fillBox)),
  arrow: wrapcommand('arrow', 0,
  ["<u>arrow(color, size)</u> Draws an arrow. " +
      "<mark>arrow red, 100</mark>"], animatedDotCommand(fillArrow)),
  mirror: wrapcommand('mirror', 1,
  ["<u>mirror(flipped)</u> Mirrors the turtle across its main axis, or " +
      "unmirrors if flipped if false. " +
      "<mark>mirror(true)</mark>"],
  function mirror(cc, val) {
    this.plan(function(j, elem) {
      cc.appear(j);
      var c = $.map($.css(elem, 'turtleScale').split(' '), parseFloat);
      if (c.length === 1) { c.push(c[0]); }
      if ((c[0] * c[1] < 0) === (!val)) {
        c[0] = -c[0];
        this.css('turtleScale', c.join(' '));
      }
      cc.resolve(j);
    });
    return this;
  }),
  twist: wrapcommand('twist', 1,
  ["<u>twist(degrees)</u> Set the primary direction of the turtle. Allows " +
      "use of images that face a different direction than 'up': " +
      "<mark>twist(-90)</mark>"],
  function twist(cc, val) {
    this.plan(function(j, elem) {
      cc.appear(j);
      if ($.isWindow(elem) || elem.nodeType === 9) return;
      this.css('turtleTwist', val);
      cc.resolve(j);
    });
    return this;
  }),
  scale: wrapcommand('scale', 1,
  ["<u>scale(factor)</u> Scales all motion up or down by a factor. " +
      "To double all drawing: <mark>scale(2)</mark>"], scaleCmd),
  grow: wrapcommand('grow', 1,
  ["<u>grow(factor)</u> Changes the size of the element by a factor. " +
      "To double the size: <mark>grow(2)</mark>"], grow),
  pause: wrapcommand('pause', 1,
  ["<u>pause(seconds)</u> Pauses some seconds before proceeding. " +
      "<mark>fd 100; pause 2.5; bk 100</mark>",
   "<u>pause(turtle)</u> Waits for other turtles to be done before " +
      "proceeding. <mark>t = new Turtle().fd 100; pause t; bk 100</mark>"],
  function pause(cc, seconds) {
    var qname = 'fx', promise = null, completion = null;
    if (seconds && $.isFunction(seconds.done)) {
      // Accept a promise-like object that has a "done" method.
      promise = seconds;
      completion = seconds.done;
    } else if ($.isFunction(seconds)) {
      // Or accept a function that is assumed to take a callback.
      completion = seconds;
    }
    if (completion) {
      this.queue(function() {
        var elem = this;
        completion.call(promise, function() {
          // If the user sets up a callback that is called more than
          // once, then ignore calls after the first one.
          var once = elem;
          elem = null;
          if (once) { $.dequeue(once); }
        });
      });
    } else {
      // Pause for some number of seconds.
      this.plan(function(j, elem) {
        cc.appear(j);
        visiblePause(elem, seconds);
        this.plan(cc.resolver(j));
      });
    }
    return this;
  }),
  st: wrapcommand('st', 0,
  ["<u>st()</u> Show turtle. The reverse of " +
      "<u>ht()</u>. <mark>do st</mark>"],
  function st(cc) {
    this.plan(function(j) {
      cc.appear(j);
      this.show();
      cc.resolve(j);
    });
    return this;
  }),
  ht: wrapcommand('ht', 0,
  ["<u>ht()</u> Hide turtle. The turtle can be shown again with " +
      "<u>st()</u>. <mark>do ht</mark>"],
  function ht(cc) {
    this.plan(function(j) {
      cc.appear(j);
      this.hide();
      cc.resolve(j);
    });
    return this;
  }),
  pu:
  function pu() {
    return this.pen(false, continuationArg(arguments, 0));
  },
  pd:
  function pd() {
    return this.pen(true, continuationArg(arguments, 0));
  },
  pe:
  function pe() {
    return this.pen('erase', continuationArg(arguments, 0));
  },
  pf:
  function pf() {
    return this.pen('path', continuationArg(arguments, 0));
  },
  clip: wrapcommand('clip', 1,
  ["<u>Clips tranparent bits out of the image of the sprite, " +
      "and sets the hit region."],
  function clip(cc, threshold) {
    if (threshold == null) {
      threshold = 0.125;
    }
    return this.plan(function(j, elem) {
      cc.appear(j);
      if (elem.tagName == 'CANVAS') {
        var hull = transparentHull(elem, threshold),
            sel = $(elem),
            origin = readTransformOrigin(elem);
        eraseOutsideHull(elem, hull);
        scalePolygon(hull,
          parseFloat(sel.css('width')) / elem.width,
          parseFloat(sel.css('height')) / elem.height,
          -origin[0], -origin[1]);
        sel.css('turtleHull', hull);
      }
      cc.resolve(j);
    });
  }),
  say: wrapcommand('say', 1,
  ["<u>say(words)</u> Say something. Use English words." +
      "<mark>say \"Let's go!\"</mark>"],
  function say(cc, words) {
    this.plan(function(j, elem) {
      cc.appear(j);
      this.queue(function(next) {
        utterSpeech(words, function() {
          cc.resolve(j);
          next();
        });
      });
    });
    return this;
  }),
  play: wrapcommand('play', 1,
  ["<u>play(notes)</u> Play notes. Notes are specified in " +
      "<a href=\"http://abcnotation.com/\" target=\"_blank\">" +
      "ABC notation</a>.  " +
      "<mark>play \"de[dBFA]2[cGEC]4\"</mark>"],
  function play(cc, notes) {
    this.plan(function(j, elem) {
      cc.appear(j);
      this.queue(function(next) {
        var instrument = getTurtleInstrument(elem),
            args = $.makeArray(cc.args),
            dowait = true,
            continuation = function() { cc.resolve(j); next(); };
        if (args.length > 0 && $.isPlainObject(args[0]) &&
            args[0].hasOwnProperty('wait')) {
          dowait = args[0].wait;
        }
        if (dowait) { args.push(continuation); }
        instrument.play.apply(instrument, args);
        if (!dowait) { continuation(); }
      });
    });
    return this;
  }),
  tone: wrapraw('tone',
  ["<u>tone(freq)</u> Immediately sound a tone. " +
      "<u>tone(freq, 0)</u> Stop sounding the tone. " +
      "<u>tone(freq, v, secs)</u> Play a tone with a volume and duration. " +
      "Frequency may be a number in Hz or a letter pitch. " +
      "<mark>tone 440, 5</mark>"],
  function tone(freq, secs) {
    var args = arguments;
    return this.each(function(j, elem) {
      var instrument = getTurtleInstrument(elem);
      instrument.tone.apply(instrument, args);
    });
  }),
  silence: wrapraw('silence',
  ["<u>silence()</u> immediately silences sound from play() or tone()."],
  function silence() {
    return this.each(function(j, elem) {
      var instrument = getTurtleInstrument(elem);
      instrument.silence();
    });
  }),
  speed: wrapcommand('speed', 1,
  ["<u>speed(persec)</u> Set one turtle's speed in moves per second: " +
      "<mark>turtle.speed 60</mark>"],
  function speed(cc, mps) {
    this.plan(function(j, elem) {
      cc.appear(j);
      this.css('turtleSpeed', mps);
      this.plan(function() {
        cc.resolve(j);
      });
    });
    return this;
  }),
  wear: wrapcommand('wear', 1,
  ["<u>wear(color)</u> Sets the turtle shell color: " +
      "<mark>wear turquoise</mark>",
      // Deal with "tan" and "fill".
   "<u>wear(url)</u> Sets the turtle image url: " +
      "<mark>wear 'http://bit.ly/1bgrQ0p'</mark>"],
  function wear(cc, name, css) {
    if ((typeof(name) == 'object' || typeof(name) == 'number') &&
         typeof(css) == 'string') {
      var t = css;
      css = name;
      name = t;
    }
    if (typeof(css) == 'number') {
      css = { height: css };
    }
    var img = nameToImg(name, 'turtle'), intick = insidetick;
    if (!img) return this;
    if (css) {
      $.extend(img.css, css);
    }
    this.plan(function(j, elem) {
      cc.appear(j);
      // Bug workaround - if background isn't cleared early enough,
      // the turtle image doesn't update.  (Even though this is done
      // later in applyImg.)
      this.css({
        backgroundImage: 'none',
      });
      var loaded = false, waiting = null;
      applyImg(this, img, function() {
        loaded = true;
        var callback = waiting;
        if (callback) { waiting = null; callback(); }
      });
      if (!canMoveInstantly(this)) {
        this.delay(animTime(elem, intick));
      }
      if (!loaded) {
        this.pause({done: function(cb) {
          if (loaded) { cb(); } else { waiting = cb; }
        }});
      }
      this.plan(function() {
        cc.resolve(j);
      });
    });
    return this;
  }),
  saveimg: wrapcommand('saveimg', 1,
  ["<u>saveimg(filename)</u> Saves the turtle's image as a file. " +
      "<mark>t.saveimg 'mypicture.png'</mark>"],
  function saveimg(cc, filename) {
    return this.plan(function(j, elem) {
      cc.appear(j);
      var ok = false;
      if (!filename) { filename = 'img'; }
      var canvas = this.canvas();
      if (!canvas) {
        see.html('<span style="color:red">Cannot saveimg: not a canvas</span>');
      } else {
        var dataurl = canvas.toDataURL();
        var dparts = /^data:image\/(\w+);base64,(.*)$/i.exec(dataurl);
        if (!dparts) {
          see.html('<span style="color:red">Cannot saveimg: ' +
              'canvas toDataURL did not work as expected.</span>');
        } else {
          if (dparts[1] && filename.toLowerCase().lastIndexOf(
                '.' + dparts[1].toLowerCase()) !=
                    Math.max(0, filename.length - dparts[1].length - 1)) {
            filename += '.' + dparts[1];
          }
          ok = true;
          dollar_turtle_methods.save(filename, atob(dparts[2]), function() {
            cc.resolve(j);
          });
        }
      }
      if (!ok) {
        cc.resolve(j);
      }
    });
  }),
  drawon: wrapcommand('drawon', 1,
  ["<u>drawon(canvas)</u> Switches to drawing on the specified canvas. " +
      "<mark>A = new Sprite('100x100'); " +
      "drawon A; pen red; fd 50; done -> A.rt 360</mark>"],
  function drawon(cc, canvas) {
    this.each(function() {
      var state = getTurtleData(this);
      if (state.drawOnCanvasSync) sync(this, state.drawOnCanvasSync);
      state.drawOnCanvasSync = canvas;
    });
    sync(canvas, this);
    return this.plan(function(j, elem) {
      cc.appear(j);
      var state = getTurtleData(elem);
      if (!canvas || canvas === global) {
        state.drawOnCanvas = null;
      } else if (canvas.jquery && $.isFunction(canvas.canvas)) {
        state.drawOnCanvas = canvas.canvas();
      } else if (canvas.tagName && canvas.tagName == 'CANVAS') {
        state.drawOnCanvas = canvas;
      } else if (canvas.nodeType == 1 || canvas.nodeType == 9) {
        state.drawOnCanvas = $(canvas).canvas();
      }
      cc.resolve(j);
    });
  }),
  label: wrapcommand('label', 1,
  ["<u>label(text)</u> Labels the current position with HTML: " +
      "<mark>label 'remember'</mark>",
   "<u>label(text, styles, labelsite)</u> Optional position specifies " +
      "'top', 'bottom', 'left', 'right', and optional styles is a size " +
      "or CSS object: " +
      "<mark>label 'big', { color: red, fontSize: 100 }, 'bottom'</mark>"],
  function label(cc, html, side, styles) {
    if ((!styles || typeof(styles) == 'string') &&
        ($.isNumeric(side) || $.isPlainObject(side))) {
      // Handle switched second and third argument order.
      var t = styles;
      styles = side;
      side = t;
    }
    if ($.isNumeric(styles)) {
      styles = { fontSize: styles };
    }
    if (side == null) {
      side =
        styles && 'labelSide' in styles ? styles.labelSide :
        styles && 'label-side' in styles ? styles['label-side'] :
        side = 'rotated scaled';
    }
    var intick = insidetick;
    return this.plan(function(j, elem) {
      cc.appear(j);
      var applyStyles = {},
          currentStyles = this.prop('style');
      // For defaults, copy inline styles of the turtle itself except for
      // properties in the following list (these are the properties used to
      // make the turtle look like a turtle).
      for (var j2 = 0; j2 < currentStyles.length; ++j2) {
        var styleProperty = currentStyles[j2];
        if (/^(?:width|height|opacity|background-image|background-size)$/.test(
          styleProperty) || /transform/.test(styleProperty)) {
          continue;
        }
        applyStyles[$.camelCase(styleProperty)] = currentStyles[styleProperty];
      }
      // And then override turtle styles with absolute positioning; and
      // override all styles with any explicity provided styles to get
      // sizing correct.
      $.extend(applyStyles, {
        position: 'absolute',
        display: 'table',
        top: 0,
        left: 0
      }, styles);
      // Place the label on the screen using the figured styles.
      var out = prepareOutput(html, 'label').result.css(applyStyles)
          .addClass('turtlelabel').appendTo(getTurtleField());
      // If the output has a turtleinput, then forward mouse events.
      if (out.hasClass('turtleinput') || out.find('.turtleinput').length) {
        mouseSetupHook.apply(out.get(0));
      }
      if (styles && 'id' in styles) {
        out.attr('id', styles.id);
      }
      if (styles && 'class' in styles) {
        out.addClass(styles.class);
      }
      var rotated = /\brotated\b/.test(side),
          scaled = /\bscaled\b/.test(side);
      // Mimic the current position and rotation and scale of the turtle.
      out.css({
        turtlePosition: computeTargetAsTurtlePosition(
            out.get(0), this.pagexy(), null, 0, 0),
        turtleRotation: rotated ? this.css('turtleRotation') : 0,
        turtleScale: scaled ? this.css('turtleScale') : 1
      });
      var gbcr = out.get(0).getBoundingClientRect();
      // Modify top-left to slide to the given corner, if requested.
      if (/\b(?:top|bottom)\b/.test(side)) {
        applyStyles.top =
            (/\btop\b/.test(side) ? -1 : 1) * gbcr.height / 2;
      }
      if (/\b(?:left|right)\b/.test(side)) {
        applyStyles.left =
            (/\bleft\b/.test(side) ? -1 : 1) * gbcr.width / 2;
      }
      // Then finally apply styles (turtle styles may be overridden here).
      out.css(applyStyles);
      // Add a delay.
      if (!canMoveInstantly(this)) {
        this.delay(animTime(elem, intick));
      }
      this.plan(function() {
        cc.resolve(j);
      });
    });
  }),
  reload: wrapcommand('reload', 0,
  ["<u>reload()</u> Does a reload, recycling content (cycling animated gifs)."],
  function reload(cc) {
    // Used to reload images to cycle animated gifs.
    this.plan(function(j, elem) {
      cc.appear(j);
      if ($.isWindow(elem) || elem.nodeType === 9) {
        global.location.reload();
        cc.resolve(j);
        return;
      }
      if (elem.src) {
        var src = elem.src;
        elem.src = '';
        elem.src = src;
      }
      cc.resolve(j);
    });
    return this;
  }),
  hatch:
  function(count, spec) {
    if (!this.length) return;
    if (spec === undefined && !$.isNumeric(count)) {
      spec = count;
      count = 1;
    }
    // Determine the container in which to hatch the turtle.
    var container = this[0];
    if ($.isWindow(container) || container.nodeType === 9) {
      container = getTurtleField();
    } else if (/^(?:br|img|input|hr|canvas)$/i.test(container.tagName)) {
      container = container.parentElement;
    }
    // Create the turtle(s)
    if (count === 1) {
      // Pass through identical jquery instance in the 1 case.
      return hatchone(
          typeof spec === 'function' ? spec(0) : spec,
          container, 'turtle');
    } else {
      var k = 0, result = [];
      for (; k < count; ++k) {
        result.push(hatchone(
            typeof spec === 'function' ? spec(k) : spec,
            container, 'turtle')[0]);
      }
      return $(result);
    }
  },
  pagexy: wrappredicate('pagexy',
  ["<u>pagexy()</u> Page coordinates {pageX:, pageY}, top-left based: " +
      "<mark>c = pagexy(); fd 500; moveto c</mark>"],
  function pagexy() {
    if (!this.length) return;
    var internal = getCenterInPageCoordinates(this[0]);
    // Copy the instance so we don't pass a reference to a cached position.
    return { pageX: internal.pageX, pageY: internal.pageY };
  }),
  getxy: wrappredicate('getxy',
  ["<u>getxy()</u> Graphing coordinates [x, y], center-based: " +
      "<mark>v = getxy(); move -v[0], -v[1]</mark>"],
  function getxy() {
    if (!this.length) return;
    return computePositionAsLocalOffset(this[0]);
  }),
  direction: wrappredicate('direction',
  ["<u>direction()</u> Current turtle direction. North is 0; East is 90: " +
      "<mark>direction()</mark>",
   "<u>direction(obj)</u> <u>direction(x, y)</u> Returns the direction " +
      "from the turtle towards an object or coordinate. " +
      "Also see <u>turnto</u>: " +
      "<mark>direction lastclick</mark>"],
  function direction(x, y) {
    if (!this.length) return;
    var elem = this[0], pos = x, dir, cur;
    if (pos !== undefined) {
      cur = $(elem).pagexy();
      if ($.isNumeric(y) && $.isNumeric(x)) { pos = [x, y]; }
      if ($.isArray(pos)) {
        pos = convertLocalXyToPageCoordinates(elem, [pos])[0];
      }
      if (!isPageCoordinate(pos)) {
        try { pos = $(pos).pagexy(); }
        catch(e) { }
      }
      if (!pos) { return NaN; }
      return radiansToDegrees(
          Math.atan2(pos.pageX - cur.pageX, cur.pageY - pos.pageY));
    }
    if ($.isWindow(elem) || elem.nodeType === 9) return 0;
    return getDirectionOnPage(elem);
  }),
  distance: wrappredicate('distance',
  ["<u>distance(obj)</u> Returns the distance from the turtle to " +
      "another object: <mark>distance lastclick</mark>",
   "<u>distance(x, y)</u> Returns the distance from the turtle to " +
      "graphing coorindates: <mark>distance(100, 0)</mark>"],
  function distance(pos, y) {
    if (!this.length) return;
    var elem = this[0], dx, dy, cur = $(elem).pagexy();
    if ($.isNumeric(y) && $.isNumeric(pos)) { pos = [pos, y]; }
    if ($.isArray(pos)) {
      pos = convertLocalXyToPageCoordinates(elem, [pos])[0];
    }
    if (!isPageCoordinate(pos)) {
      try { pos = $(pos).pagexy(); }
      catch(e) { }
    }
    if (!pos) { return NaN; }
    dx = pos.pageX - cur.pageX;
    dy = pos.pageY - cur.pageY;
    return Math.sqrt(dx * dx + dy * dy);
  }),
  canvas: wrapraw('canvas',
  ["<u>turtle.canvas()</u> The canvas for the turtle image. " +
      "Draw on the turtle: " +
      "<mark>c = turtle.canvas().getContext('2d'); c.fillStyle = red; " +
      "c.fillRect(10, 10, 30, 30)</mark>"],
  function canvas() {
    return this.filter('canvas').get(0) || this.find('canvas').get(0);
  }),
  imagedata: wrapraw('imagedata',
  ["<u>imagedata()</u> Returns the image data for the turtle. " +
      "<mark>imdat = imagedata(); write imdat.data.length, 'bytes'</mark>",
   "<u>imagedata(imdat)</u> Sets the image data for the turtle. " +
      "<mark>imagedata({width: 1, height:1, data:[255,0,0,255]});</mark>",
  ],
  function imagedata(val) {
    var canvas = this.canvas();
    if (!canvas) {
      if (val) throw new Error(
        'can only set imagedata on a canvas like a Sprite');
      var img = this.filter('img').get(0);
      if (!img) return;
      canvas = getOffscreenCanvas(img.naturalWidth, img.naturalHeight);
      canvas.getContext('2d').drawImage(img, 0, 0);
    }
    var ctx = canvas.getContext('2d');
    if (!val) {
      // The read case: return the image data for the whole canvas.
      return ctx.getImageData(0, 0, canvas.width, canvas.height);
    }
    // The write case: if it's not an ImageData, convert it to one.
    if (!(val instanceof ImageData)) {
      if (typeof val != 'object' ||
          !$.isNumeric(val.width) || !$.isNumeric(val.height) ||
          !($.isArray(val.data) || val.data instanceof Uint8ClampedArray ||
            val.data instanceof Uint8Array)) {
        return;
      }
      var imdat = ctx.createImageData(
          Math.round(val.width), Math.round(val.height));
      var minlen = Math.min(val.data.length, imdat.data.length);
      for (var j = 0; j < minlen; ++j) { imdat.data[j] = val.data[j]; }
      val = imdat;
    }
    // If the size must be changed, resize it.
    if (val.width != canvas.width ||
        val.height != canvas.height) {
      var oldOrigin = readTransformOrigin(canvas);
      canvas.width = val.width;
      canvas.height = val.height;
      var newOrigin = readTransformOrigin(canvas);
      // Preserve the origin if it's a turtle.
      moveToPreserveOrigin(canvas, oldOrigin, newOrigin);
      // Drop the turtle hull, if any.
      $(canvas).css('turtleHull', 'auto');
      ctx = canvas.getContext('2d');
    }
    // Finally put the image data into the canvas.
    ctx.putImageData(val, 0, 0);
  }),
  cell: wrapraw('cell',
  ["<u>cell(r, c)</u> Row r and column c in a table. " +
      "Use together with the table function: " +
      "<mark>g = table 8, 8; g.cell(0,2).text 'hello'</mark>"],
  function cell(r, c) {
    var sel = this.find(
        $.isNumeric(r) ? 'tr:nth-of-type(' + (r + 1) + ')' : 'tr');
    return sel.find(
        $.isNumeric(c) ? 'td:nth-of-type(' + (c + 1) + ')' : 'td');
  }),
  shown: wrappredicate('shown',
  ["<u>shown()</u> True if turtle is shown, false if hidden: " +
      "<mark>do ht; write shown()</mark>"],
  function shown() {
    var elem = this.get(0);
    return elem && !invisible(elem);
  }),
  hidden: wrappredicate('hidden',
  ["<u>hidden()</u> True if turtle is hidden: " +
      "<mark>do ht; write hidden()</mark>"],
  function hidden() {
    var elem = this.get(0);
    return !elem || invisible(elem);
  }),
  inside: wrappredicate('inside',
  ["<u>inside(obj)</u> True if the turtle is encircled by obj: " +
      "<mark>inside(window)</mark>"],
  function inside(elem) {
    if (!elem) return false;
    if (typeof elem == 'string') {
      elem = $(elem);
    }
    if (elem.jquery) {
      if (!elem.length || invisible(elem[0])) return false;
      elem = elem[0];
    }
    var gbcr0 = getPageGbcr(elem),
        encloser = null, rectenc = false,
        allok = true, j = 0, k, obj;
    for (; allok && j < this.length; ++j) {
      obj = this[j];
      // Optimize the outside-bounding-box case.
      if (isDisjointGbcr(gbcr0, getPageGbcr(obj))) {
        return false;
      }
      if (!encloser) {
        encloser = getCornersInPageCoordinates(elem);
        rectenc = polyMatchesGbcr(encloser, gbcr0);
      }
      // Optimize the rectilinear-encloser case.
      if (rectenc && gbcrEncloses(gbcr0, getPageGbcr(obj))) {
        continue;
      }
      if (isPageCoordinate(obj)) {
        allok &= pointInConvexPolygon(obj, encloser);
      } else {
        allok &= doesConvexPolygonContain(
          encloser, getCornersInPageCoordinates(obj));
      }
    }
    return !!allok;
  }),
  touches: wrappredicate('touches',
  ["<u>touches(obj)</u> True if the turtle touches obj: " +
      "<mark>touches(lastclick)</mark>",
   "<u>touches(color)</u> True if the turtle touches a drawn color: " +
      "<mark>touches red</mark>"],
  function touches(arg, y) {
    if (!this.length || invisible(this[0])) { return false; }
    if (typeof(arg) == "function" && isCSSColor(arg.helpname)) {
      arg = arg.helpname;
    }
    if (arg == 'color' || isCSSColor(arg)) {
      return touchesPixel(this[0], arg == 'color' ? null : arg);
    }
    if ($.isNumeric(arg) && $.isNumeric(y)) {
      arg = [arg, y];
    }
    if ($.isArray(arg) && arg.length == 2 &&
        $.isNumeric(arg[0]) && $.isNumeric(arg[1])) {
      arg = convertLocalXyToPageCoordinates(this[0] || document.body, [arg])[0];
    }
    if (!arg) return false;
    if (typeof arg === 'string') { arg = $(arg); }
    if (!arg.jquery && !$.isArray(arg)) { arg = [arg]; }
    var anyok = false, k = 0, j, obj, elem, gbcr0, toucher, gbcr1;
    for (;!anyok && k < this.length; ++k) {
      elem = this[k];
      gbcr0 = getPageGbcr(elem);
      // hidden elements do not touch anything
      if (gbcr0.width == 0) { continue; }
      toucher = null;
      for (j = 0; !anyok && j < arg.length; ++j) {
        obj = arg[j];
        // Optimize the outside-bounding-box case.
        gbcr1 = getPageGbcr(obj);
        if (isDisjointGbcr(gbcr0, gbcr1)) {
          continue;
        }
        // Do not touch removed or hidden elements, or points without
        // a pageX/pageY coordinate.
        if (gbcr1.width == 0 && (obj.pageX == null || obj.pageY == null)) {
          continue;
        }
        if (!toucher) {
          toucher = getCornersInPageCoordinates(elem);
        }
        if (isPageCoordinate(obj)) {
          anyok |= pointInConvexPolygon(obj, toucher);
        } else {
          anyok |= doConvexPolygonsOverlap(
            toucher, getCornersInPageCoordinates(obj));
        }
      }
    }
    return !!anyok;
  }),
  within: wrappredicate('within',
  ["<u>within(distance, obj)</u> Filters elements to those " +
      "within distance of obj: " +
      "<mark>$('.turtle').within(100, lastclick)</mark>"],
  function within(distance, x, y) {
    return withinOrNot(this, true, distance, x, y);
  }),
  notwithin: wrappredicate('notwithin',
  ["<u>within(distance, obj)</u> Filters elements to those " +
      "further than distance of obj: " +
      "<mark>$('.turtle').notwithin(100, lastclick)</mark>"],
  function notwithin(distance, x, y) {
    return withinOrNot(this, false, distance, x, y);
  }),
  nearest: wrappredicate('nearest',
  ["<u>nearest(obj)</u> Filters elements to those nearest obj" +
      "<mark>$('.turtle').neareest(lastclick)</mark>"],
  function nearest(x, y) {
    var pos, result = [], mind2 = Infinity, gbcr, j;
    if ($.isNumeric(pos) && $.isNumeric(y)) {
      pos = [x, y];
    } else {
      pos = x;
    }
    if ($.isArray(pos)) {
      // [x, y]: local coordinates.
      pos = convertLocalXyToPageCoordinates(this[0] || document.body, [pos])[0];
    }
    if (!isPageCoordinate(pos)) {
      try { pos = $(pos).pagexy(); }
      catch(e) { pos = null; }
    }
    for (j = 0; j < this.length; j++) {
      gbcr = getPageGbcr(this[j]);
      if (!result.length || !isGbcrOutside(pos, mind2, gbcr)) {
        var thispos = getCenterInPageCoordinates(this[j]),
            dx = pos.pageX - thispos.pageX,
            dy = pos.pageY - thispos.pageY,
            d2 = dx * dx + dy * dy;
        if (d2 <= mind2) {
          if (d2 < mind2) {
            mind2 = d2;
            result.length = 0;
          }
          result.push(this[j]);
        }
      }
    }
    return $(result);
  }),
  done: wrapraw('done',
  ["<u>done(fn)</u> Calls fn when animation is complete. Use with await: " +
      "<mark>await done defer()</mark>"],
  function done(callback) {
    var sync = this;
    return this.promise().done(function() {
      if (sync) {
        // Never do callback synchronously.  Instead redo the promise
        // callback after a zero setTimeout.
        var async = sync;
        async_pending += 1;
        setTimeout(function() {
          async_pending -= 1;
          async.promise().done(callback);
        }, 0);
      } else {
        callback.apply(this, arguments);
      }
    });
  }),
  plan: wrapraw('plan',
  ["<u>plan(fn)</u> Runs fn in the animation queue. For planning logic: " +
      "<mark>write getxy(); fd 50; plan -> write getxy(); bk 50"],
  function plan(qname, callback, args) {
    if ($.isFunction(qname)) {
      args = callback;
      callback = qname;
      qname = 'fx';
    }
    // If animation is active, then plan will queue the callback.
    // It will also arrange things so that if the callback enqueues
    // further animations, they are inserted at the same location,
    // so that the callback can expand into several animations,
    // just as an ordinary function call expands into its subcalls.
    function enqueue(elem, index, elemqueue) {
      var action = (args ?
            (function() { callback.apply($(elem), args); }) :
            (function() { callback.call($(elem), index, elem); })),
          lastanim = elemqueue.length && elemqueue[elemqueue.length - 1],
          animation = (function() {
            var saved = $.queue(this, qname),
                subst = [], inserted;
            if (saved[0] === 'inprogress') {
              subst.unshift(saved.shift());
            }
            $.queue(elem, qname, subst);
            action();
            // The Array.prototype.push is faster.
            // $.merge($.queue(elem, qname), saved);
            Array.prototype.push.apply($.queue(elem, qname), saved);
            nonrecursive_dequeue(elem, qname);
          });
      animation.finish = action;
      $.queue(elem, qname, animation);
    }
    var elem, sel, length = this.length, j = 0;
    for (; j < length; ++j) {
      elem = this[j];
      // Special case: first wait for an unloaded image to load.
      queueWaitIfLoadingImg(elem, qname);
      // Queue an animation if there is a queue.
      var elemqueue = $.queue(elem, qname);
      if (elemqueue.length) {
        enqueue(elem, j, elemqueue);
      } else if (args) {
        callback.apply($(elem), args);
      } else {
        callback.call($(elem), j, elem);
      }
    }
    return this;
  })
};

//////////////////////////////////////////////////////////////////////////
// QUEUING SUPPORT
//////////////////////////////////////////////////////////////////////////

function queueShowHideToggle() {
  $.each(['toggle', 'show', 'hide'], function(i, name) {


    var builtInFn = $.fn[name];
    // Change show/hide/toggle to queue their behavior by default.
    // Since animating show/hide will call the zero-argument
    // form synchronously at the end of animation, we avoid
    // infinite recursion by examining jQuery's internal fxshow
    // state and avoiding the recursion if the animation is calling
    // show/hide.
    $.fn[name] = function(speed, easing, callback) {
      var a = arguments;
      // TODO: file a bug in jQuery to allow solving this without _data.
      if (!a.length && this.hasClass('turtle') &&
          (this.length > 1 || !$._data(this[0], 'fxshow'))) {
        a = [0];
      }
      builtInFn.apply(this, a);
    }
  });
}

// If the queue for an image is empty, starts by queuing a wait-for-load.
function queueWaitIfLoadingImg(img, qname) {
  if (!qname) qname = 'fx';
  if (img.tagName == 'IMG' && img.src && !img.complete) {
    var queue = $.queue(img, qname);
    if (queue.length == 0) {
      $.queue(img, qname, function(next) {
        afterImageLoadOrError(img, null, next);
      });
      nonrecursive_dequeue(img, qname);
    }
  }
}

//////////////////////////////////////////////////////////////////////////
// HUNG LOOP DETECTION
//////////////////////////////////////////////////////////////////////////

var warning_shown = {},
    loopCounter = 0,
    hungTimer = null,
    hangStartTime = null;

// When a student makes an infinite loop of turtle motions, the turtle
// will not move "right away" as it might have on an old Apple II;
// instead it will just hang the tab.  So to detect this error, this
// function function fires off a zero-timeout setTimeout message every
// 100th turtle motion.  If it takes more than a few seconds to receive it,
// our script is blocking message dispatch, and an interrupt is triggered.
function checkForHungLoop(fname) {
  if ($.turtle.hangtime == Infinity || loopCounter++ < 100) {
    return;
  }
  loopCounter = 0;
  var now = (new Date).getTime();
  if (!hangStartTime) {
    hangStartTime = now;
    clearTimeout(hungTimer);
    hungTimer = setTimeout(function() {
      clearTimeout(hungTimer);
      hungTimer = null;
      hangStartTime = null;
    }, 0);
    return;
  }
  // Timeout after which we interrupt the program.
  if (now - hangStartTime > $.turtle.hangtime) {
    if (see.visible()) {
      see.html('<span style="color:red">Oops: program ' +
        'interrupted because it was hanging the browser. ' +
        'Try reducing the number of repetitions.  Or try using ' +
        '<b style="background:yellow">await done defer()</b> or ' +
        '<b style="background:yellow">tick</b> ' +
        'to make an animation.</span>');
    }
    $.turtle.interrupt('hung');
  }
}


// It is unreasonable (and a common error) to queue up motions to try to
// change the value of a predicate.  The problem is that queuing will not
// do anything immediately.  This check prints a warning and flushes the
// queue when the queue is 100 long.
function checkPredicate(fname, sel) {
  if ($.turtle.nowarn) return;
  var ok = true, j;
  for (j = 0; ok && j < sel.length; ++j) {
    if ($.queue(sel[j]).length >= 100) {
      ok = false;
    }
  }
  if (!ok) {
    if (!warning_shown[fname]) {
      warning_shown[fname] = 1;
      if (see.visible()) {
        see.html('<span style="color:red">Oops: ' + fname +
        ' may not return useful results when motion is queued. ' +
        'Try <b style="background:yellow">speed Infinity</b></span> or ' +
        '<b style="background:yellow">await done defer()</b> first.');
      } else {
        console.warn(fname + ' may not return useful results when motion ' +
        'is queued.  Try "speed Infinity" or "await done defer()".');
      }
    }
    sel.finish();
  }
}

// LEGACY NAMES

function deprecate(map, oldname, newname) {
  map[oldname] = function() {
    if (!(oldname in warning_shown)) {
      see.html('<span style="color:red;">' + oldname + ' deprecated.  Use ' +
          newname + '.</span>');
      warning_shown[oldname] = 1;
    }
    // map[oldname] = map[newname];
    return map[newname].apply(this, arguments);
  }
  if (map[newname].__super__) {
    // Handle legacy class names by extending the correct class.
    __extends(map[oldname], map[newname]);
  }
}
deprecate(turtlefn, 'move', 'slide');
deprecate(turtlefn, 'direct', 'plan');
deprecate(turtlefn, 'enclosedby', 'inside');
deprecate(turtlefn, 'bearing', 'direction');

$.fn.extend(turtlefn);

//////////////////////////////////////////////////////////////////////////
// TURTLE GLOBAL ENVIRONMENT
// Implements educational support when $.turtle() is called:
// * Looks for an element #id to use as the turtle (id defaults to 'turtle').
// * If not found, does a hatch(id).
// * Turns every #id into a global variable.
// * Sets up globals for "lastclick", "lastmousemove" etc.
// * Sets up global functions for all turtle functions for the main turtle.
// * Sets up a global "tick" function.
// * Sets up a global "speed" function and does a speed(10) by default.
// * Sets up a global "hatch" function to make a new turtle.
//////////////////////////////////////////////////////////////////////////

var turtleGIFUrl = "data:image/gif;base64,R0lGODlhKAAwAPIFAAAAAAFsOACSRTCuSICAgP///wAAAAAAACH5BAlkAAYAIf8LTkVUU0NBUEUyLjADAQAAACwAAAAAKAAwAAAD72i6zATEgBCAebHpzUnxhDAMAvhxKOoV3ziuZyo3RO26dTbvgXj/gsCO9ysOhENZz+gKJmcUkmA6PSKfSqrWieVtuU+KGNXbXofLEZgR/VHCgdua4isGz9mbmM6U7/94BmlyfUZ1fhqDhYuGgYqMkCOBgo+RfWsNlZZ3ewIpcZaIYaF6XaCkR6aokqqrk0qrqVinpK+fsbZkuK2ouRy0ob4bwJbCibthh6GYebGcY7/EsWqTbdNG1dd9jnXPyk2d38y0Z9Yub2yA6AvWPYk+zEnkv6xdCoPuw/X2gLqy9vJIGAN4b8pAgpQOIlzI8EkCACH5BAlkAAYALAAAAAAoADAAAAPuaLrMBMSAEIB5senNSfGEMAwC+HEo6hXfOK5nKjdE7bp1Nu+BeP+CwI73Kw6EQ1nP6AomZxSSYDo9Ip9KqtaJ5W25Xej3qqGYsdEfZbMcgZXtYpActzLMeLOP6c7f3nVNfEZ7TXSFg4lyZAYBio+LZYiQfHMbc3iTlG9ilGpdjp4ujESiI6RQpqegqkesqqhKrbEpoaa0KLaiuBy6nrxss6+3w7tomo+cDXmBnsoLza2nsb7SN2tl1nyozVOZTJhxysxnd9XYCrrAtT7KQaPruavBo2HQ8xrvffaN+GV5/JbE45fOG8Ek5Q4qXHgwAQA7";

var eventfn = { click:1, dblclick:1, mouseup:1, mousedown:1, mousemove:1 };

function global_turtle_animating() {
  return (global_turtle && $.queue(global_turtle).length > 0);
}

var global_turtle = null;
var global_turtle_methods = [];
var attaching_ids = false;
var dollar_turtle_methods = {
  interrupt: wrapraw('interrupt',
  ["<u>interrupt()</u> Interrupts and aborts all turtle commands."],
  function interrupt(option) {
    if (option == 'reset') {
      // interrupt('reset') resets a flag that re-enables turtle
      // commands.  Until a reset, all turtle commands will throw
      // exceptions after an interrupt().
      interrupted = false;
      return;
    }
    if (option == 'test') {
      // interrupt('test') returns true if something is running that is
      // interruptable by interrupt().  It is used by the IDE to determine
      // if a "stop" button should be shown.
      if (interrupted) return false;
      if (tickinterval) return true;
      if ($.timers.length) return true;
      if (forever_timers.length) return true;
      if (async_pending) return true;
      if (global_turtle_animating()) return true;
      if ($(':animated').length) return true;
      if ($('.turtle').filter(function() {
        return $.queue(this).length > 0; }).length > 0) return true;
      if ($('.turtleinput').filter(function() {
        return !$(this).prop('disabled')}).length > 0) return true;
      if (windowhasturtleevent()) return true;
      return false;
    }
    // Stop all animations.
    $(':animated,.turtle').clearQueue().stop();
    // Stop our audio.
    resetAudio();
    // Disable all input.
    $('.turtleinput').prop('disabled', true);
    // Detach all event handlers on the window.
    $(global).off('.turtleevent');
    // Low-level detach all jQuery events
    $('*').not('#_testpanel *').map(
       function(i, e) { $._data(e, 'events', null) });
    // Set a flag that will cause all commands to throw.
    interrupted = true;
    // Turn off the global tick interval timer.
    globaltick(null, null);
    // Turn off timers for 'forever'
    clearForever();
    // Run through any remaining timers, stopping each one.
    // This handles the case of animations (like "dot") that
    // are not attached to an HTML element.
    for (var j = $.timers.length - 1; j >= 0; --j) {
      if ($.timers[j].anim && $.timers[j].anim.elem) {
        $($.timers[j].anim.elem).stop(true, true);
      }
    }
    // Throw an interrupt exception.
    var msg = option ? "'" + option + "'" : '';
    throw new Error('interrupt(' + msg + ') called');
  }),
  cs: wrapglobalcommand('cs',
  ["<u>cs()</u> Clear screen. Erases both graphics canvas and " +
      "body text: <mark>do cs</mark>"],
  function cs() {
    clearField();
  }),
  cg: wrapglobalcommand('cg',
  ["<u>cg()</u> Clear graphics. Does not alter body text: " +
      "<mark>do cg</mark>"],
  function cg() {
    clearField('canvas labels');
  }),
  ct: wrapglobalcommand('ct',
  ["<u>ct()</u> Clear text. Does not alter graphics canvas: " +
      "<mark>do ct</mark>"],
  function ct() {
    clearField('text');
  }),
  canvas: wrapraw('canvas',
  ["<u>canvas()</u> Returns the raw turtle canvas. " +
      "<mark>c = canvas().getContext('2d'); c.fillStyle = red; " +
      "c.fillRect(100,100,200,200)</mark>"],
  function canvas() {
    return getTurtleDrawingCanvas();
  }),
  sizexy: wrapraw('sizexy',
  ["<u>sizexy()</u> Get the document pixel [width, height]. " +
      "<mark>[w, h] = sizexy(); canvas('2d').fillRect(0, 0, w, h)</mark>"],
  sizexy),
  forever: wrapraw('forever',
  ["<u>forever(fn)</u> Calls fn repeatedly, forever. " +
      "<mark>forever -> fd 2; rt 2</mark>",
   "<u>forever(fps, fn)</u> Calls fn repeating fps per second. " +
      "<mark>forever 2, -> fd 25; dot blue</mark>"],
  forever),
  stop: wrapraw('stop',
  ["<u>stop()</u> stops the current forever loop. " +
      "<mark>forever -> fd 10; if not inside window then stop()</mark>",
   "<u>stop(fn)</u> stops the forever loop corresponding to fn.",
   "Use <u>break</u> to stop a <u>for</u> or <u>while</u> loop."],
  stop),
  tick: wrapraw('tick',
  ["<u>tick(fps, fn)</u> Calls fn fps times per second until " +
      "<u>tick</u> is called again: " +
      "<mark>c = 10; tick 1, -> c and write(c--) or tick()</mark>"],
  function tick(tps, fn) {
    if (global_turtle_animating()) {
      var sel = $(global_turtle);
      sel.plan(function() {
        globaltick(tps, fn);
      });
    } else {
      globaltick(tps, fn);
    }
  }),
  speed: wrapglobalcommand('speed',
  ["<u>speed(mps)</u> Sets default turtle speed in moves per second: " +
      "<mark>speed Infinity</mark>"],
  function globalspeed(mps) {
    globaldefaultspeed(mps);
  }),
  say: wrapraw('say',
  ["<u>say(words)</u> Say something. Use English words." +
      "<mark>say \"Let's go!\"</mark>"],
  function say(words) {
    if (global_turtle) {
      var sel = $(global_turtle);
      sel.say.call(sel, words);
    } else {
      var cc = setupContinuation(null, 'say', arguments, 0);
      cc.appear(null);
      utterSpeech(words, function() { cc.resolve(null); });
      cc.exit();
    }
  }),
  play: wrapraw('play',
  ["<u>play(notes)</u> Play notes. Notes are specified in " +
      "<a href=\"http://abcnotation.com/\" target=\"_blank\">" +
      "ABC notation</a>.  " +
      "<mark>play \"de[dBFA]2[cGEC]4\"</mark>"],
  function play() {
    if (global_turtle) {
      var sel = $(global_turtle);
      sel.play.apply(sel, arguments);
    } else {
      var cc = setupContinuation(null, 'play', arguments, 0);
      cc.appear(null);
      var instrument = getGlobalInstrument(),
          args = $.makeArray(cc.args);
      args.push(function() { cc.resolve(null); });
      instrument.play.apply(instrument, args);
      cc.exit();
    }
  }),
  tone: wrapraw('tone',
  ["<u>tone(freq)</u> Immediately sound a tone. " +
      "<u>tone(freq, 0)</u> Stop sounding the tone. " +
      "<u>tone(freq, v, secs)</u> Play a tone with a volume and duration. " +
      "Frequency may be a number in Hz or a letter pitch. " +
      "<mark>tone 440, 5</mark>"],
  function tone() {
    if (global_turtle) {
      var sel = $(global_turtle);
      sel.tone.apply(sel, arguments);
    } else {
      var instrument = getGlobalInstrument();
      instrument.play.apply(instrument);
    }
  }),
  silence: wrapraw('silence',
  ["<u>silence()</u> Immediately silences sound from play() or tone()."],
  function silence() {
    if (global_turtle) {
      var sel = $(global_turtle);
      sel.silence();
    } else {
      var instrument = getGlobalInstrument();
      instrument.silence();
    }
  }),
  sync: wrapraw('sync',
  ["<u>sync(t1, t2, t3,...)</u> " +
      "Selected turtles wait for each other to stop."], sync),
  remove: wrapraw('remove',
  ["<u>remove(t)</u> " +
      "Remove selected turtles."], remove),
  done: wrapraw('done',
  ["<u>done(fn)</u> Calls fn when animation is complete. Use with await: " +
      "<mark>await done defer()</mark>"],
  function done(callback) {
    var sync = $('.turtle');
    return sync.promise().done(function() {
      if (sync) {
        // Never do callback synchronously.  Instead redo the promise
        // callback after a zero setTimeout.
        var async = sync;
        async_pending += 1;
        setTimeout(function() {
          async_pending -= 1;
          async.promise().done(callback);
        }, 0);
      } else {
        callback.apply(this, arguments);
      }
    });
  }),
  load: wrapraw('load',
  ["<u>load(url, cb)</u> Loads data from the url and passes it to cb. " +
      "<mark>load 'intro', (t) -> write 'intro contains', t</mark>"],
  function(url, cb) {
    var val;
    $.ajax(apiUrl(url, 'load'), { async: !!cb, complete: function(xhr) {
      try {
        val = xhr.responseObject = JSON.parse(xhr.responseText);
        if (typeof(val.data) == 'string' && typeof(val.file) == 'string') {
          val = val.data;
          if (/\.json(?:$|\?|\#)/.test(url)) {
            try { val = JSON.parse(val); } catch(e) {}
          }
        } else if ($.isArray(val.list) && typeof(val.directory) == 'string') {
          val = val.list;
        } else if (val.error) {
          val = null;
        }
      } catch(e) {
        if (val == null && xhr && xhr.responseText) {
          val = xhr.responseText;
        }
      }
      if (cb) {
        cb(val, xhr);
      }
    }});
    return val;
  }),
  save: wrapraw('save',
  ["<u>save(url, data, cb)</u> Posts data to the url and calls when done. " +
      "<mark>save 'intro', 'pen gold, 20\\nfd 100\\n'</mark>"],
  function(url, data, cb) {
    if (!url) throw new Error('Missing url for save');
    var payload = { }, key;
    url = apiUrl(url, 'save');
    if (/\.json(?:$|\?|\#)/.test(url)) {
      data = JSON.stringify(data, null, 2);
    }
    if (typeof(data) == 'string' || typeof(data) == 'number') {
      payload.data = data;
    } else {
      for (key in data) if (data.hasOwnProperty(key)) {
        if (typeof data[key] == 'string') {
          payload[key] = data[key];
        } else {
          payload[key] = JSON.stringify(data[key]);
        }
      }
    }
    if (payload && !payload.key) {
      var login = loginCookie();
      if (login && login.key && login.user == pencilUserFromUrl(url)) {
        payload.key = login.key;
      }
    }
    $.ajax(apiUrl(url, 'save'), {
      type: 'POST',
      data: payload,
      complete: function(xhr) {
        var val
        try {
          val = JSON.parse(xhr.responseText);
        } catch(e) {
          if (val == null && xhr && xhr.responseText) {
            val = xhr.responseText;
          }
        }
        if (cb) {
          cb(val, xhr);
        }
      }
    });
  }),
  append: wrapglobalcommand('append',
  ["<u>append(html)</u> Appends text to the document without a new line. " +
      "<mark>append 'try this twice...'</mark>"],
  function append(html) {
    $.fn.append.apply($('body'), arguments);
  }),
  type: wrapglobalcommand('type',
  ["<u>type(text)</u> Types preformatted text like a typewriter. " +
      "<mark>type 'Hello!\n'</mark>"], plainTextPrint),
  typebox: wrapglobalcommand('typebox',
  ["<u>typebox(clr)</u> Draws a colored box as typewriter output. " +
      "<mark>typebox red</mark>"], function(c, t) {
    if (t == null && c != null && !isCSSColor(c)) { t = c; c = null; }
    plainBoxPrint(c, t);
  }),
  typeline: wrapglobalcommand('typebox',
  ["<u>typeline()</u> Same as type '\\n'. " +
      "<mark>typeline()</mark>"], function(t) {
    plainTextPrint((t || '') + '\n');
  }),
  write: wrapglobalcommand('write',
  ["<u>write(html)</u> Writes a line of text. Arbitrary HTML may be written: " +
      "<mark>write 'Hello, world!'</mark>"], doOutput, function() {
    return prepareOutput(Array.prototype.join.call(arguments, ' '), 'div');
  }),
  read: wrapglobalcommand('read',
  ["<u>read(fn)</u> Reads text or numeric input. " +
      "Calls fn once: " +
      "<mark>read (x) -> write x</mark>",
   "<u>read(html, fn)</u> Prompts for input: " +
      "<mark>read 'Your name?', (v) -> write 'Hello ' + v</mark>"],
  doOutput, function read(a, b) { return prepareInput(a, b, 0); }),
  readnum: wrapglobalcommand('readnum',
  ["<u>readnum(html, fn)</u> Reads numeric input. Only numbers allowed: " +
      "<mark>readnum 'Amount?', (v) -> write 'Tip: ' + (0.15 * v)</mark>"],
  doOutput, function readnum(a, b) { return prepareInput(a, b, 'number'); }),
  readstr: wrapglobalcommand('readstr',
  ["<u>readstr(html, fn)</u> Reads text input. Never " +
      "converts input to a number: " +
      "<mark>readstr 'Enter code', (v) -> write v.length + ' long'</mark>"],
  doOutput, function readstr(a, b) { return prepareInput(a, b, 'text'); }),
  listen: wrapglobalcommand('listen',
  ["<u>listen(html, fn)</u> Reads voice input, if the browser supports it:" +
      "<mark>listen 'Say something', (v) -> write v</mark>"],
  doOutput, function readstr(a, b) { return prepareInput(a, b, 'voice'); }),
  menu: wrapglobalcommand('menu',
  ["<u>menu(map)</u> shows a menu of choices and calls a function " +
      "based on the user's choice: " +
      "<mark>menu {A: (-> write 'chose A'), B: (-> write 'chose B')}</mark>"],
  doOutput, prepareMenu),
  button: wrapglobalcommand('button',
  ["<u>button(text, fn)</u> Writes a button. Calls " +
      "fn whenever the button is clicked: " +
      "<mark>button 'GO', -> fd 100</mark>"],
  doOutput, prepareButton),
  table: wrapglobalcommand('table',
  ["<u>table(m, n)</u> Writes m rows and c columns. " +
      "Access cells using <u>cell</u>: " +
      "<mark>g = table 8, 8; g.cell(2,3).text 'hello'</mark>",
   "<u>table(array)</u> Writes tabular data. " +
      "Each nested array is a row: " +
      "<mark>table [[1,2,3],[4,5,6]]</mark>"],
  doOutput, prepareTable),
  img: wrapglobalcommand('img',
  ["<u>img(url)</u> Writes an image with the given address. " +
      "Any URL can be provided.  A name without slashes will be " +
      "treated as '/img/name'." +
      "<mark>t = img 'tree'</mark>"],
  doOutput, prepareImage),
  random: wrapraw('random',
  ["<u>random(n)</u> Random non-negative integer less than n: " +
      "<mark>write random 10</mark>",
   "<u>random(list)</u> Random member of the list: " +
      "<mark>write random ['a', 'b', 'c']</mark>",
   "<u>random('position')</u> Random page position: " +
      "<mark>moveto random 'position'</mark>",
   "<u>random('color')</u> Random color: " +
      "<mark>pen random 'color'</mark>"],
  random),
  rgb: wrapraw('rgb',
  ["<u>rgb(r,g,b)</u> Makes a color out of red, green, and blue parts. " +
      "<mark>pen rgb(150,88,255)</mark>"],
  function(r, g, b) { return componentColor('rgb', [
      Math.max(0, Math.min(255, Math.floor(r))),
      Math.max(0, Math.min(255, Math.floor(g))),
      Math.max(0, Math.min(255, Math.floor(b))) ]); }),
  hatch: // Deprecated - no docs.
  function hatch(count, spec) {
    return $(document).hatch(count, spec);
  },
  rgba: wrapraw('rgba',
  ["<u>rgba(r,g,b,a)</u> Makes a color out of red, green, blue, and alpha. " +
      "<mark>pen rgba(150,88,255,0.5)</mark>"],
  function(r, g, b, a) { return componentColor('rgba', [
      Math.max(0, Math.min(255, Math.floor(r))),
      Math.max(0, Math.min(255, Math.floor(g))),
      Math.max(0, Math.min(255, Math.floor(b))),
      a ]); }),
  hsl: wrapraw('hsl',
  ["<u>hsl(h,s,l)</u> Makes a color out of hue, saturation, and lightness. " +
      "<mark>pen hsl(120,0.65,0.75)</mark>"],
  function(h, s, l) { return componentColor('hsl', [
     h,
     (s * 100).toFixed(0) + '%',
     (l * 100).toFixed() + '%']); }),
  hsla: wrapraw('hsla',
  ["<u>hsla(h,s,l,a)</u> Makes a color out of hue, saturation, lightness, " +
      "alpha. <mark>pen hsla(120,0.65,0.75,0.5)</mark>"],
  function(h, s, l, a) { return componentColor('hsla', [
     h,
     (s * 100).toFixed(0) + '%',
     (l * 100).toFixed(0) + '%',
     a]); }),
  click: wrapwindowevent('click',
  ["<u>click(fn)</u> Calls fn(event) whenever the mouse is clicked. " +
      "<mark>click (e) -> moveto e; label 'clicked'</mark>"]),
  dblclick: wrapwindowevent('dblclick',
  ["<u>dblclick(fn)</u> Calls fn(event) whenever the mouse is double-clicked. " +
      "<mark>dblclick (e) -> moveto e; label 'double'</mark>"]),
  mouseup: wrapwindowevent('mouseup',
  ["<u>mouseup(fn)</u> Calls fn(event) whenever the mouse is released. " +
      "<mark>mouseup (e) -> moveto e; label 'up'</mark>"]),
  mousedown: wrapwindowevent('mousedown',
  ["<u>mousedown(fn)</u> Calls fn(event) whenever the mouse is pressed. " +
      "<mark>mousedown (e) -> moveto e; label 'down'</mark>"]),
  mousemove: wrapwindowevent('mousemove',
  ["<u>mousemove(fn)</u> Calls fn(event) whenever the mouse is moved. " +
      "<mark>mousemove (e) -> write 'at ', e.x, ',', e.y</mark>"]),
  keydown: wrapwindowevent('keydown',
  ["<u>keydown(fn)</u> Calls fn(event) whenever a key is pushed down. " +
      "<mark>keydown (e) -> write 'down ' + e.key</mark>"]),
  keyup: wrapwindowevent('keyup',
  ["<u>keyup(fn)</u> Calls fn(event) whenever a key is released. " +
      "<mark>keyup (e) -> write 'up ' + e.key</mark>"]),
  keypress: wrapwindowevent('keypress',
  ["<u>keypress(fn)</u> Calls fn(event) whenever a character key is pressed. " +
      "<mark>keypress (e) -> write 'press ' + e.key</mark>"]),
  send: wrapraw('send',
  ["<u>send(name)</u> Sends a message to be received by recv. " +
      "<mark>send 'go'; recv 'go', -> fd 100</mark>"],
  function send(name) {
    var args = arguments;
    var message = Array.prototype.slice.call(args, 1),
        sq = sendRecvData.sent[name];
    if (!sq) { sq = sendRecvData.sent[name] = []; }
    sq.push(message);
    pollSendRecv(name);
  }),
  recv: wrapraw('recv',
  ["<u>recv(name, fn)</u> Calls fn once when a sent message is received. " +
      "<mark>recv 'go', (-> fd 100); send 'go'</mark>"],
  function recv(name, cb) {
    var wq = sendRecvData.waiting[name];
    if (!wq) { wq = sendRecvData.waiting[name] = []; }
    wq.push(cb);
    pollSendRecv(name);
  }),
  abs: wrapraw('abs',
  ["<u>abs(x)</u> The absolute value of x. " +
      "<mark>see abs -5</mark>"], Math.abs),
  acos: wrapraw('acos',
  ["<u>acos(x)</u> Trigonometric arccosine, in radians. " +
      "<mark>see acos 0.5</mark>"], Math.acos),
  asin: wrapraw('asin',
  ["<u>asin(y)</u> Trigonometric arcsine, in radians. " +
      "<mark>see asin 0.5</mark>"], Math.asin),
  atan: wrapraw('atan',
  ["<u>atan(y, x = 1)</u> Trigonometric arctangent, in radians. " +
      "<mark>see atan 0.5</mark>"],
  function atan(y, x) { return Math.atan2(y, (x == undefined) ? 1 : x); }
  ),
  cos: wrapraw('cos',
  ["<u>cos(radians)</u> Trigonometric cosine, in radians. " +
      "<mark>see cos 0</mark>"], Math.cos),
  sin: wrapraw('sin',
  ["<u>sin(radians)</u> Trigonometric sine, in radians. " +
      "<mark>see sin 0</mark>"], Math.sin),
  tan: wrapraw('tan',
  ["<u>tan(radians)</u> Trigonometric tangent, in radians. " +
      "<mark>see tan 0</mark>"], Math.tan),

  // For degree versions of trig functions, make sure we return exact
  // results when possible. The set of values we have to consider is
  // fortunately very limited. See "Rational Values of Trigonometric
  // Functions." http://www.jstor.org/stable/2304540

  acosd: wrapraw('acosd',
  ["<u>acosd(x)</u> Trigonometric arccosine, in degrees. " +
      "<mark>see acosd 0.5</mark>"],
   function acosd(x) {
     switch (x) {
       case   1: return   0;
       case  .5: return  60;
       case   0: return  90;
       case -.5: return 120;
       case  -1: return 180;
     }
     return Math.acos(x) * 180 / Math.PI;
  }),
  asind: wrapraw('asind',
  ["<u>asind(x)</u> Trigonometric arcsine, in degrees. " +
      "<mark>see asind 0.5</mark>"],
  function asind(x) {
    switch (x) {
      case   1: return  90;
      case  .5: return  30;
      case   0: return   0;
      case -.5: return -30;
      case  -1: return -90;
    }
    return Math.asin(x) * 180 / Math.PI;
  }),
  atand: wrapraw('atand',
  ["<u>atand(y, x = 1)</u> Trigonometric arctangent, " +
      "in degrees. <mark>see atand -1, 0/mark>"],
  function atand(y, x) {
    if (x == undefined) { x = 1; }
    if (y == 0) {
      return (x == 0) ? NaN : ((x > 0) ? 0 : 180);
    } else if (x == 0) {
      return (y > 0) ? Infinity : -Infinity;
    } else if (Math.abs(y) == Math.abs(x)) {
      return (y > 0) ? ((x > 0) ? 45 : 135) :
                       ((x > 0) ? -45 : -135);
    }
    return Math.atan2(y, x) * 180 / Math.PI;
  }),
  cosd: wrapraw('cosd',
  ["<u>cosd(degrees)</u> Trigonometric cosine, in degrees. " +
      "<mark>see cosd 45</mark>"],
  function cosd(x) {
    x = modulo(x, 360);
    if (x % 30 === 0) {
      switch ((x < 0) ? x + 360 : x) {
        case   0: return   1;
        case  60: return  .5;
        case  90: return   0;
        case 120: return -.5;
        case 180: return  -1;
        case 240: return -.5;
        case 270: return   0;
        case 300: return  .5;
      }
    }
    return Math.cos(x / 180 * Math.PI);
  }),
  sind: wrapraw('sind',
  ["<u>sind(degrees)</u> Trigonometric sine, in degrees. " +
      "<mark>see sind 45</mark>"],
  function sind(x) {
    x = modulo(x, 360);
    if (x % 30 === 0) {
      switch ((x < 0) ? x + 360 : x) {
        case   0: return   0;
        case  30: return  .5;
        case  90: return   1;
        case 150: return  .5;
        case 180: return   0;
        case 210: return -.5;
        case 270: return  -1;
        case 330: return -.5;
      }
    }
    return Math.sin(x / 180 * Math.PI);
  }),
  tand: wrapraw('tand',
  ["<u>tand(degrees)</u> Trigonometric tangent, in degrees. " +
      "<mark>see tand 45</mark>"],
  function tand(x) {
    x = modulo(x, 360);
    if (x % 45 === 0) {
      switch ((x < 0) ? x + 360 : x) {
        case   0: return 0;
        case  45: return 1;
        case  90: return Infinity;
        case 135: return -1;
        case 180: return 0;
        case 225: return 1;
        case 270: return -Infinity;
        case 315: return -1
      }
    }
    return Math.tan(x / 180 * Math.PI);
  }),
  ceil: wrapraw('ceil',
  ["<u>ceil(x)</u> Round up. " +
      "<mark>see ceil 1.9</mark>"], Math.ceil),
  floor: wrapraw('floor',
  ["<u>floor(x)</u> Round down. " +
      "<mark>see floor 1.9</mark>"], Math.floor),
  round: wrapraw('round',
  ["<u>round(x)</u> Round to the nearest integer. " +
      "<mark>see round 1.9</mark>"], Math.round),
  exp: wrapraw('exp',
  ["<u>exp(x)</u> Raise e to the power x. " +
      "<mark>see exp 2</mark>"], Math.exp),
  ln: wrapraw('ln',
  ["<u>ln(x)</u> The natural logarithm of x. " +
      "<mark>see ln 2</mark>"], Math.log),
  log10: wrapraw('log10',
  ["<u>log10(x)</u> The base 10 logarithm of x. " +
      "<mark>see log10 0.01</mark>"],
  function log10(x) { return roundEpsilon(Math.log(x) * Math.LOG10E); }),
  pow: wrapraw('pow',
  ["<u>pow(x, y)</u> Raise x to the power y. " +
      "<mark>see pow 4, 1.5</mark>"],
  function pow(x, y) { return roundEpsilon(Math.pow(x, y)); }),
  sqrt: wrapraw('sqrt',
  ["<u>sqrt(x)</u> The square root of x. " +
      "<mark>see sqrt 25</mark>"], Math.sqrt),
  max: wrapraw('max',
  ["<u>max(x, y, ...)</u> The maximum of a set of values. " +
      "<mark>see max -5, 2, 1</mark>"], Math.max),
  min: wrapraw('min',
  ["<u>min(x, y, ...)</u> The minimum of a set of values. " +
      "<mark>see min 2, -5, 1</mark>"], Math.min),
  Pencil: wrapraw('Pencil',
  ["<u>new Pencil(canvas)</u> " +
      "Make an invisble pencil for drawing on a canvas. " +
      "<mark>s = new Sprite; p = new Pencil(s); " +
      "p.pen red; p.fd 100; remove p</mark>"],
      Pencil),
  Turtle: wrapraw('Turtle',
  ["<u>new Turtle(color)</u> Make a new turtle. " +
      "<mark>t = new Turtle; t.fd 100</mark>"], Turtle),
  Piano: wrapraw('Piano',
  ["<u>new Piano(keys)</u> Make a new piano. " +
      "<mark>t = new Piano 88; t.play 'edcdeee'</mark>"], Piano),
  Webcam: wrapraw('Webcam',
  ["<u>new Webcam(options)</u> Make a new webcam. " +
      "<mark>v = new Webcam; v.plan -> pic = new Sprite v</mark>"],
  Webcam),
  Sprite: wrapraw('Sprite',
  ["<u>new Sprite({width:w,height:h,color:c})</u> " +
      "Make a new sprite to <mark>drawon</mark>. " +
      "<mark>s = new Sprite({width:50,height:50,color:blue}); " +
      "s.fd 100</mark>"], Sprite),
  loadscript: wrapraw('loadscript',
  ["<u>loadscript(url, callback)</u> Loads Javascript or Coffeescript from " +
       "the given URL, calling callback when done."],
  function loadscript(url, callback) {
    if (global.CoffeeScript && /\.(?:coffee|cs)$/.test(url)) {
      CoffeeScript.load(url, callback);
    } else {
      $.getScript(url, callback);
    }
  }),
  pressed: wrapraw('pressed',
  ["<u>pressed('control')</u> Tests if a specific key is pressed. " +
      "<mark>if pressed 'a' then write 'a was pressed'</mark>",
   "<u>pressed.list()</u> Returns a list of pressed keys, by name. " +
      "<mark>write 'Pressed keys: ' + pressed.list().join(',')</mark>"
  ], pressedKey),
  help: globalhelp
};

var extrahelp = {
  finish: {helptext: ["<u>finish()</u> Finishes turtle animation. " +
      "Does not pause for effect: " +
      "<mark>do finish</mark>"]}
};

var sendRecvData = {
  // message passing support
  sent: {},
  waiting: {},
  pollTimer: null
};

function pollSendRecv(name) {
  if (sendRecvData.pollTimer === null) {
    var sq = sendRecvData.sent[name],
        wq = sendRecvData.waiting[name];
    if (wq && wq.length && sq && sq.length) {
      sendRecvData.pollTimer = setTimeout(function() {
        sendRecvData.pollTimer = null;
        if (wq && wq.length && sq && sq.length) {
          wq.shift().apply(null, sq.shift())
          pollSendRecv(name);
        }
      }, 0);
    }
  }
}


deprecate(dollar_turtle_methods, 'defaultspeed', 'speed');

dollar_turtle_methods.save.loginCookie = loginCookie;

var helpok = {};

var colors = [
  "aliceblue", "antiquewhite", "aqua", "aquamarine", "azure", "beige",
  "bisque", "black", "blanchedalmond", "blue", "blueviolet", "brown",
  "burlywood", "cadetblue", "chartreuse", "chocolate", "coral",
  "cornflowerblue", "cornsilk", "crimson", "cyan", "darkblue", "darkcyan",
  "darkgoldenrod", "darkgray", "darkgrey", "darkgreen", "darkkhaki",
  "darkmagenta", "darkolivegreen", "darkorange", "darkorchid", "darkred",
  "darksalmon", "darkseagreen", "darkslateblue", "darkslategray",
  "darkslategrey", "darkturquoise", "darkviolet", "deeppink", "deepskyblue",
  "dimgray", "dimgrey", "dodgerblue", "firebrick", "floralwhite",
  "forestgreen", "fuchsia", "gainsboro", "ghostwhite", "gold", "goldenrod",
  "gray", "grey", "green", "greenyellow", "honeydew", "hotpink",
  "indianred", "indigo", "ivory", "khaki", "lavender", "lavenderblush",
  "lawngreen", "lemonchiffon", "lightblue", "lightcoral", "lightcyan",
  "lightgoldenrodyellow", "lightgray", "lightgrey", "lightgreen",
  "lightpink", "lightsalmon", "lightseagreen", "lightskyblue",
  "lightslategray", "lightslategrey", "lightsteelblue", "lightyellow",
  "lime", "limegreen", "linen", "magenta", "maroon", "mediumaquamarine",
  "mediumblue", "mediumorchid", "mediumpurple", "mediumseagreen",
  "mediumslateblue", "mediumspringgreen", "mediumturquoise",
  "mediumvioletred", "midnightblue", "mintcream", "mistyrose", "moccasin",
  "navajowhite", "navy", "oldlace", "olive", "olivedrab", "orange",
  "orangered", "orchid", "palegoldenrod", "palegreen", "paleturquoise",
  "palevioletred", "papayawhip", "peachpuff", "peru", "pink", "plum",
  "powderblue", "purple", "rebeccapurple", "red", "rosybrown", "royalblue",
  "saddlebrown", "salmon", "sandybrown", "seagreen", "seashell", "sienna",
  "silver", "skyblue", "slateblue", "slategray", "slategrey", "snow",
  "springgreen", "steelblue", "tan", "teal", "thistle", "tomato",
  "turquoise", "violet", "wheat", "white", "whitesmoke", "yellow",
  "yellowgreen", "transparent"
];

(function() {
  var specialstrings = [
    "none", "erase", "path", "up", "down",  // Pen modes.
    "color", "position", "normal", // Random modes.
    "touch" // Special Within distances.
  ];
  var definedstrings = specialstrings.concat(colors), j = 0;
  for (; j < definedstrings.length; j++) {
    if (!dollar_turtle_methods.hasOwnProperty(definedstrings[j])) {
      dollar_turtle_methods[definedstrings[j]] = definedstrings[j];
    }
  }
  dollar_turtle_methods.PI = Math.PI;
  dollar_turtle_methods.E = Math.E;
  dollar_turtle_methods.print = dollar_turtle_methods.write
  extrahelp.colors = {helptext:
      ["Defined colors: " + colors.join(" ")]};
  extrahelp.see = {helptext:
      ["<u>see(v)</u> Shows the value of v in the test panel: " +
      "<mark>see document</mark>"]};
  extrahelp.if = extrahelp.else = extrahelp.then = {helptext:
      ["<u>if</u> <u>then</u> <u>else</u> Tests a condition: " +
      "<mark>if 1 <= (new Date).getDay() <= 5 then " +
      "write 'Working hard!' else write 'Happy weekend!'</mark>"]};
  extrahelp.await = extrahelp.defer = {helptext:
      ["<u>await</u> <u>defer</u> Waits for results from an " +
       "asynchronous event; from " +
       '<a href="http://maxtaco.github.io/coffee-script/" target="_blank"' +
       ">Iced CoffeeScript</a>: " +
       "<mark>await readnum defer n</mark>"]};
})();

$.turtle = function turtle(id, options) {
  var exportedsee = false;
  if (arguments.length == 1 && typeof(id) == 'object' && id &&
      !id.hasOwnProperty('length')) {
    options = id;
    id = 'turtle';
  }
  id = id || 'turtle';
  options = options || {};
  if ('turtle' in options) {
    id = options.turtle;
  }
  // Clear any previous turtle methods.
  clearGlobalTurtle();
  // Expand any <script type="text/html"> unless htmlscript is false.
  // This is to simplify literal HTML editing within templated editors.
  if (!('htmlscript' in options) || options.htmlscript) {
    $('script[type="text/html"]').each(function() {
        $(this).replaceWith(
            $(this).html().replace(/^\x3c!\[CDATA\[\n?|\]\]\x3e$/g, ''));
    });
  }
  if (!globalDrawing.ctx && ('subpixel' in options)) {
    globalDrawing.subpixel = parseInt(options.subpixel);
  }
  // Set up hung-browser timeout, default 20 seconds.
  $.turtle.hangtime = ('hangtime' in options) ?
      parseFloat(options.hangtime) : 20000;

  // Set up global events.
  if (options.events !== false) {
    turtleevents(options.eventprefix);
  }
  if (options.pressed !== false) {
    addKeyEventHooks();
    pressedKey.enable(true);
  }
  // Set up global log function.
  if (options.see !== false) {
    exportsee();
    exportedsee = true;
    if (global.addEventListener) {
      global.addEventListener('error', see);
    } else {
      global.onerror = see;
    }
    // Set up an alias.
    global.debug = see;
    // 'debug' should be used now instead of log
    deprecate(global, 'log', 'debug');
  }
  if (options.queuehide !== false) {
    queueShowHideToggle();
  }

  // Copy $.turtle.* functions into global namespace.
  if (options.functions !== false) {
    global.printpage = global.print;
    $.extend(global, dollar_turtle_methods);
  }
  // Set default turtle speed
  globaldefaultspeed(('defaultspeed' in options) ?
      options.defaultspeed : 1);
  // Initialize audio context (avoids delay in first notes).
  if (isAudioPresent()) try {
    getAudioTop();
  } catch (e) { }
  // Find or create a singleton turtle if one does not exist.
  var selector = null;
  var wrotebody = false;
  if (id) {
    selector = $('#' + id);
    if (!selector.length) {
      if (!$('body').length) {
        // Initializing without a body?  Force one in!
        document.write('<body>');
        wrotebody = true;
      }
      selector = new Turtle(id);
    }
  }
  if (selector && !selector.length) { selector = null; }
  // Globalize selected jQuery methods of a singleton turtle.
  if (selector && selector.length === 1 && (options.global !== false)) {
    var extraturtlefn = {
      css:1, fadeIn:1, fadeOut:1, fadeTo:1, fadeToggle:1,
      animate:1, toggle:1, finish:1, promise:1, direct:1,
      show:1, hide:1 };
    var globalfn = $.extend({}, turtlefn, extraturtlefn);
    global_turtle_methods.push.apply(global_turtle_methods,
       globalizeMethods(selector, globalfn));
    global_turtle = selector[0];
    // Make sure the main turtle is visible over other normal sprites.
    selector.css({zIndex: 1});
  }
  // Set up global objects by id.
  if (options.ids !== false) {
    turtleids(options.idprefix);
    if (selector && id) {
      global[id] = selector;
    }
  }
  // Set up test console.
  if (options.panel !== false) {
    var seeopt = {
      title: 'test panel (type help for help)',
      abbreviate: [undefined, helpok],
      consolehook: seehelphook
    };
    if (selector) { seeopt.abbreviate.push(selector); }
    if (options.title) {
      seeopt.title = options.title;
    }
    if (options.panelheight) {
      seeopt.height = options.panelheight;
    }
    see.init(seeopt);
    // Return an eval loop hook string if 'see' is exported.
    if (exportedsee) {
      if (global.CoffeeScript) {
        return "see.init(eval(see.cs))";
      } else {
        return see.here;
      }
    }
  }
  return $('#' + id);
};

$.extend($.turtle, dollar_turtle_methods);
$.turtle.colors = colors;

function seehelphook(text, result) {
  // Also, check the command to look for (non-CoffeeScript) help requests.
  if ((typeof result == 'function' || typeof result == 'undefined')
      && /^\w+\s*$/.test(text)) {
    if (result && result.helptext) {
      globalhelp(result);
      return true;
    } else if (text in extrahelp) {
      globalhelp(text);
      return true;
    }
  } else if (typeof result == 'undefined' && /^help\s+\S+$/.test(text)) {
    globalhelp(/^help\s+(\S+)$/.exec(text)[1]);
    return true;
  }
  return false;
}

function copyhelp(method, fname, extrahelp, globalfn) {
  if (method.helptext) {
    globalfn.helptext = method.helptext;
  } else if (fname in extrahelp) {
    globalfn.helptext = extrahelp[fname].helptext;
  }
  globalfn.method = method;
  globalfn.helpname = fname;
  return globalfn;
}

function globalizeMethods(thisobj, fnames) {
  var replaced = [];
  for (var fname in fnames) {
    if (fnames.hasOwnProperty(fname) && !(fname in global)) {
      replaced.push(fname);
      global[fname] = (function(fname) {
        var method = thisobj[fname], target = thisobj;
        return copyhelp(method, fname, extrahelp,
            (function globalized() { /* Use parentheses to call a function */
                return method.apply(target, arguments); }));
      })(fname);
    }
  }
  return replaced;
}

function clearGlobalTurtle() {
  global_turtle = null;
  for (var j = 0; j < global_turtle_methods.length; ++j) {
    delete global[global_turtle_methods[j]];
  }
  global_turtle_methods.length = 0;
}

// Hook jQuery cleanData.
var oldCleanData = $.cleanData;
$.cleanData = function(elems) {
  for (var i = 0, elem; (elem = elems[i]) !== undefined; i++) {
    // Clean up media stream.
    var state = $.data(elem, 'turtleData');
    if (state && state.stream) {
      state.stream.stop();
    }
    // Undefine global variablelem.
    if (elem.id && global[elem.id] && global[elem.id].jquery &&
        global[elem.id].length === 1 &&
        global[elem.id][0] === elem) {
      delete global[elem.id];
    }
    // Clear global turtlelem.
    if (elem === global_turtle) {
      clearGlobalTurtle();
    }
  }
}

function isCSSColor(color) {
  return rgbaForColor(color) !== null;
}

var colorCache = {};

function isNamedColor(name) {
  if (!/^[a-z]+$/.test(name)) {
    return false;
  }
  for (var j = 0; j < colors.length; ++j) {
    if (colors[j] == name) return true;
  }
  return false;
}

function rgbaForColor(color) {
  if (color in colorCache) {
    return colorCache[color];
  }
  if (!color || (!isNamedColor(color) &&
      !/^(?:rgb|hsl)a?\([^)]*\)$|^\#[a-f0-9]{3}(?:[a-f0-9]{3})?$/i.test(
          color))) {
    return null;
  }
  var d = document.createElement('div'), unset = d.style.color,
      result = null, m;
  d.style.color = color;
  if (unset !== d.style.color) {
    m = /rgba?\s*\(\s*(\d+),\s*(\d+),\s*(\d+)(?:,\s*([.\d]+))?\s*\)/.exec($(d).
        css({position:'absolute',top:0,left:0}).appendTo('body').css('color'));
    if (m) {
      result = [parseInt(m[1]), parseInt(m[2]), parseInt(m[3]),
                Math.round(255 * (m[4] ? parseFloat(m[4]) : 1))];
    } else if (color == 'transparent') {
      // IE does not convert 'transparent' to rgba.
      result = [0, 0, 0, 0];
    }
    $(d).remove();
  }
  colorCache[color] = result;
  return result;
}

function createTurtleShellOfColor(color) {
  var c = getOffscreenCanvas(40, 48);
  var ctx = c.getContext('2d'),
      cx = 20,
      cy = 26;
  ctx.beginPath();
  ctx.arc(cx, cy, 16, 0, 2 * Math.PI, false);
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
  ctx.beginPath();
  // Half of a symmetric turtle shell pattern.
  var pattern = [
    [[5, -14], [3, -11]],
    [[3, -11], [7, -8], [4, -4]],
    [[4, -4], [7, 0], [4, 4]],
    [[4, 4], [7, 8], [3, 11]],
    [[7, -8], [12, -9], null],
    [[7, 0], [15, 0], null],
    [[7, 8], [12, 9], null],
    [[3, 11], [1, 15], null]
  ];
  for (var j = 0; j < pattern.length; j++) {
    var path = pattern[j], connect = true;
    ctx.moveTo(cx + path[0][0], cy + path[0][1]);
    for (var k = 1; k < path.length; k++) {
      if (path[k] !== null) {
        ctx.lineTo(cx + path[k][0], cy + path[k][1]);
      }
    }
    for (var k = path.length - 1; k >= 0; k--) {
      if (path[k] === null) {
        k--;
        ctx.moveTo(cx - path[k][0], cy + path[k][1]);
      } else {
        ctx.lineTo(cx - path[k][0], cy + path[k][1]);
      }
    }
  }
  ctx.lineWidth = 1.1;
  ctx.strokeStyle = 'rgba(255,255,255,0.75)';
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(cx, cy, 15.5, 0, 2 * Math.PI, false);
  ctx.closePath();
  ctx.strokeStyle = 'rgba(0,0,0,0.4)';
  ctx.stroke();
  return c.toDataURL();
}

function createPointerOfColor(color) {
  var c = getOffscreenCanvas(40, 48);
  var ctx = c.getContext('2d');
  ctx.beginPath();
  ctx.moveTo(0,48);
  ctx.lineTo(20,0);
  ctx.lineTo(40,48);
  ctx.lineTo(20,36);
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
  return c.toDataURL();
}

function createRadiusOfColor(color) {
  var c = getOffscreenCanvas(40, 40);
  var ctx = c.getContext('2d');
  ctx.beginPath();
  ctx.arc(20,20,18,-5 * Math.PI / 2,-Math.PI / 2);
  ctx.closePath();
  ctx.lineTo(20, 20);
  ctx.stroke();
  ctx.strokeStyle = color;
  ctx.lineWidth = 4
  ctx.stroke();
  return c.toDataURL();
}

function createDotOfColor(color, diam) {
  diam = diam || 12;
  var c = getOffscreenCanvas(diam, diam);
  var ctx = c.getContext('2d');
  var r = diam / 2;
  ctx.beginPath();
  ctx.arc(r, r, r, 0, 2 * Math.PI);
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
  return c.toDataURL();
}

function createPencilOfColor(color) {
  var c = getOffscreenCanvas(40, 48);
  var ctx = c.getContext('2d');
  ctx.beginPath();
  function tip() {
    ctx.moveTo(19.5, 43);
    ctx.lineTo(20.5, 43);
    ctx.lineTo(21.5, 43.5);
    ctx.lineTo(25.5, 36.2);
    ctx.lineTo(24, 35.5);
    ctx.lineTo(23, 35.5);
    ctx.lineTo(20.5, 36.5);
    ctx.lineTo(19.5, 36.5);
    ctx.lineTo(17, 35.5);
    ctx.lineTo(16, 35.5);
    ctx.lineTo(14.5, 36.2);
    ctx.lineTo(18.5, 43.5);
    ctx.closePath();
  }
  tip();
  ctx.fillStyle="#ffcb6b";
  ctx.fill();
  ctx.beginPath()
  function eraser() {
    ctx.moveTo(25.5, 12);
    ctx.lineTo(25.5, 8);
    ctx.lineTo(14.5, 8);
    ctx.lineTo(14.5, 12);
    ctx.closePath();
  }
  eraser();
  ctx.fillStyle="#d1ebff";
  ctx.fill();
  ctx.beginPath();
  ctx.moveTo(19.5,48);
  ctx.lineTo(13,36);
  ctx.lineTo(13,1);
  ctx.lineTo(14,0);
  ctx.lineTo(26,0);
  ctx.lineTo(27,1);
  ctx.lineTo(27,36);
  ctx.lineTo(20.5,48);
  ctx.closePath();
  tip();
  ctx.moveTo(25.5, 12);
  ctx.lineTo(25.5, 8);
  ctx.lineTo(14.5, 8);
  ctx.lineTo(14.5, 12);
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
  return c.toDataURL();
}

// The shapes below can be used as turtle shapes using the 'wear' command.
// For example, wear("blue pencil") will render a pencil in blue, and
// then set it as the turtle shape.  In addition to an image URL,
// a turtle shape needs several other properties, namely a
// transformOrigin (center for rotation and drawing) as well as a
// turtleHull (points for a convex hull for hit-testing).
var shapes = {
  turtle: function(color) {
    if (!color) { color = 'mediumseagreen'; }
    return {
      url: createTurtleShellOfColor(color),
      css: {
        width: 20,
        height: 24,
        transformOrigin: '10px 13px',
        turtleHull: "-8 -5 -8 6 -2 -13 2 -13 8 6 8 -5 0 9",
        opacity: 0.67,
        backgroundImage: 'url(' + turtleGIFUrl + ')',
        backgroundSize: 'cover'
      }
    };
  },
  pointer: function(color) {
    if (!color) { color = 'gray'; }
    return {
      url: createPointerOfColor(color),
      css: {
        width: 20,
        height: 24,
        transformOrigin: '10px 18px',
        turtleHull: "-10 6 0 -18 10 6",
        opacity: 0.67
      }
    };
  },
  radius: function(color) {
    if (!color) { color = 'gray'; }
    return {
      url: createRadiusOfColor(color),
      css: {
        width: 20,
        height: 20,
        transformOrigin: '10px 10px',
        turtleHull: "-10 0 -7 7 0 10 7 7 10 0 7 -7 0 -10 -7 -7",
        opacity: 1
      }
    };
  },
  dot: function(color) {
    if (!color) { color = 'black'; }
    return {
      url: createDotOfColor(color, 24),
      css: {
        width: 12,
        height: 12,
        transformOrigin: '6px 6px',
        turtleHull: "-6 0 -4 4 0 6 4 4 6 0 4 -4 0 -6 -4 -4",
        opacity: 1
      }
    };
  },
  point: function(color) {
    if (!color) { color = 'black'; }
    return {
      url: createDotOfColor(color, 6),
      css: {
        width: 3,
        height: 3,
        transformOrigin: '1.5px 1.5px',
        turtleHull: "-1.5 0 -1 1 0 1.5 1 1 1.5 0 1 -1 0 -1.5 -1 -1",
        opacity: 1
      }
    };
  },
  pencil: function(color) {
    if (!color) { color = 'dodgerblue'; }
    return {
      url: createPencilOfColor(color),
      css: {
        width: 20,
        height: 24,
        transformOrigin: '10px 24px',
        turtleHull: "0 0 -3 -6 -3 -24 3 -6 3 -24",
        opacity: 1
      }
    };
  }
};

function createRectangleShape(width, height, subpixels) {
  if (!subpixels) {
    subpixels = 1;
  }
  return (function(color) {
    var c = document.createElement('canvas');
    c.width = width;
    c.height = height;
    var ctx = c.getContext('2d');
    if (!color) {
      color = "rgba(128,128,128,0.125)";
    }
    if (color != 'transparent') {
      ctx.fillStyle = color;
      ctx.fillRect(0, 0, width, height);
    }
    var sw = width / subpixels, sh = height / subpixels;
    var css = {
        width: sw,
        height: sh,
        transformOrigin: (sw / 2) + 'px + ' + (sh / 2) + 'px',
        opacity: 1
    };
    if (subpixels < 1) {
      // Requires newer than Chrome 40.
      // Avoid smooth interpolation of big pixels.
      css.imageRendering = 'pixelated';
    }
    return {
      img: c,
      css: css
    };
  });
}

function lookupShape(shapename) {
  if (!shapename) {
    return null;
  }
  if (shapename in shapes) {
    return shapes[shapename];
  }
  var m = shapename.match(/^(\d+)x(\d+)(?:\/(\d+))?$/);
  if (m) {
    return createRectangleShape(
        parseFloat(m[1]), parseFloat(m[2]), m[3] && parseFloat(m[3]));
  }
  return null;
}

function specToImage(spec, defaultshape) {
  var width = spec.width || spec.height || 256;
  var height = spec.height || spec.width || 256;
  var subpixel = spec.subpixel || 1 / (spec.scale || 1);
  var color = spec.color || 'transparent';
  var shape = createRectangleShape(width, height, subpixel);
  return shape(color);
}

function nameToImg(name, defaultshape) {
  // Parse forms for built-in shapes:
  // "red" -> red default shape (red turtle)
  // "turtle" -> default color turtle (mediumseagreen turtle)
  // "blue turtle" -> blue turtle
  // "rgba(50, 90, 255) pencil" -> bluish pencil
  // {width: 100, height: 100, color: green} -> green 100x100 square
  if (!name) { return null; }
  if ($.isPlainObject(name)) {
    return specToImage(name, defaultshape);
  }
  if ($.isFunction(name) && (name.helpname || name.name)) {
    // Deal with unquoted "tan" and "dot".
    name = name.helpname || name.name;
  }
  if (name.constructor === jQuery) {
    // Unwrap jquery objects.
    if (!name.length) { return null; }
    name = name.get(0);
  }
  if (name.tagName) {
    if (name.tagName != 'CANVAS' && name.tagName != 'IMG' &&
        name.tagName != 'VIDEO') {
      return null;
    }
    return { img: name, css: { opacity: 1 } };
  }
  var builtin = name.toString().trim().split(/\s+/),
      color = null,
      shape = null;
  if (builtin.length) {
    shape = lookupShape(builtin[builtin.length - 1]);
    if (shape) {
      builtin.pop();
    }
  }
  if (builtin.length && isCSSColor(builtin.join(' '))) {
    color = builtin.join(' ');
    builtin.length = 0;
  }
  if (!shape && color) {
    shape = lookupShape(defaultshape); // Default shape when only a color.
  }
  if (shape) {
    return shape(color);
  }
  // Default to '/img/' URLs if it doesn't match a well-known name.
  if (!/\//.test(name)) {
    name = imgUrl(name);
  }
  // Parse URLs.
  if (/\//.test(name)) {
    var hostname = absoluteUrlObject(name).hostname;
    // Use proxy to load image if the image is offdomain but the page is on
    // a pencil host (with a proxy).
    if (!isPencilHost(hostname) && isPencilHost(global.location.hostname)) {
      name = global.location.protocol + '//' +
             global.location.host + '/proxy/' + absoluteUrl(name);
    }
    return {
      url: name,
      css: {
        transformOrigin: '50% 50%',
        opacity: 1
      }
    }
  }
  return null;
}

var entityMap = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': '&quot;'
};

function escapeHtml(string) {
  if (string == null) { return ""; }
  return String(string).replace(/[&<>"]/g, function(s) {return entityMap[s];});
}

function hatchone(name, container, defaultshape) {
  var isID = name && /^[a-zA-Z]\w*$/.exec(name),
      isTag = name && /^<.*>$/.exec(name),
      img = nameToImg(name, defaultshape) ||
        (name == null) && nameToImg(defaultshape);

  // Don't overwrite previously existing id.
  if (isID && $('#' + name).length) { isID = false; }

  // Create an image element with the requested name.
  var result;
  if (isTag) {
    result = $(name);
  } else if (img) {
    result = $('<canvas>');
    applyImg(result, img);
  } else {
    result = $('<div>' + escapeHtml(name) + '</div>');
  }
  if (name && 'object' == typeof(name)) {
    if ('id' in name) {
      result.attr('id', name.id);
    }
    if ('class' in name) {
      result.addClass(name.class);
    }
  }
  // Position the turtle inside the container.
  result.css({
    position: 'absolute',
    display: 'table',
    top: 0,
    left: 0
  });
  if (!container || container.nodeType == 9 || $.isWindow(container)) {
    container = getTurtleField();
  }
  result.appendTo(container);
  // Fix top and left so that the turtle is centered with zero transform.
  var middle = readTransformOrigin(result[0]);
  result.css({
    top: -middle[1],
    left: -middle[0]
  });

  // Move it to the starting pos.
  result.css({
    turtlePosition: computeTargetAsTurtlePosition(
        result[0], $(container).pagexy(), null, 0, 0),
    turtleRotation: 0,
    turtleScale: 1});

  // Every hatched turtle has class="turtle".
  result.addClass('turtle');

  // Set the id.
  if (isID) {
    result.attr('id', name);
    // Update global variable unless there is a conflict.
    if (attaching_ids && !global.hasOwnProperty(name)) {
      global[name] = result;
    }
  }
  // Move it to the center of the document and export the name as a global.
  return result;
}

// Simplify Math.floor(Math.random() * N) and also random choice.
function random(arg, arg2) {
  if (typeof(arg) == 'number') {
    arg = Math.ceil(arg);
    if (typeof(arg2) == 'number') {
      arg2 = Math.ceil(arg2);
      return Math.floor(Math.random() * (arg2 - arg) + arg);
    }
    return Math.floor(Math.random() * arg);
  }
  if (typeof(arg) == 'object' && arg && arg.length && arg.slice) {
    return arg[Math.floor(Math.random() * arg.length)];
  }
  if (arg == 'normal') {
    // Ratio of uniforms gaussian, from tinyurl.com/9oh2nqg
    var u, v, x, y, q;
    do {
      u = Math.random();
      v = 1.7156 * (Math.random() - 0.5);
      x = u - 0.449871;
      y = Math.abs(v) + 0.386595;
      q = x * x + y * (0.19600 * y - 0.25472 * x);
    } while (q > 0.27597 && (q > 0.27846 || v * v > -4 * Math.log(u) * u * u));
    return v / u;
  }
  if (arg == 'position') {
    return {
      pageX: random(dw() + 1),
      pageY: random(dh() + 1)
    };
  }
  if (arg == 'color') {
    return 'hsl(' + Math.floor(Math.random() * 360) + ',100%,50%)';
  }
  if (arg == 'gray') {
    return 'hsl(0,0,' + Math.floor(Math.random() * 100) + '%)';
  }
  if (arg === true) {
    return Math.random() >= 0.5;
  }
  return Math.random();
}

var forever_timers = [];
var current_timers = [];

// Sets up as many as you like timers: this simplifies
// setInterval(fn, 33) to just forever(fn); it also delays
// starting the interval until the global table reaches this
// point in the animation queue.
function forever(fps, fn) {
  if (!fn && 'function' == typeof(fps)) {
    fn = fps;
    fps = 30;
  }
  var action = null;
  var ms = Math.max(Math.floor(1000 / Math.max(1/(24*60*60), fps)), 0);
  if (global_turtle_animating()) {
    var sel = $(global_turtle);
    sel.plan(function() {
      action = fn;
    });
  } else {
    action = fn;
  }
  var record = {fn: fn, timer: setInterval(function() {
    if (!action) return;
    // Set default speed to Infinity within forever().
    try {
      insidetick++;
      current_timers.push(record);
      action();
    } finally {
      insidetick--;
      current_timers.pop(record);
    }
  }, ms)};
  forever_timers.push(record);
  return record.timer;
}

// Clears forever timers matching "which", or if "which" is null,
// clears all forever timers.
function clearForever(which) {
  var cleaned = [];
  for (var j = 0; j < forever_timers.length; ++j) {
    var record = forever_timers[j];
    if (which == null || which == record.timer || which == record.fn) {
      clearInterval(forever_timers[j].timer);
    } else {
      cleaned.push(forever_timers[j]);
    }
  }
  forever_timers = cleaned;
}

// Stops the forever timer matching "which".
// If "which" is null, clears the currently-running timer, or if
// outside all forever timers, clears all forever timers.
function stop(which) {
  if (which == null && current_timers.length) {
    which = current_timers[current_timers.length - 1].timer;
  }
  clearForever(which);
}

// One-time tick: the old original one-per-program, one-per-second tick method.
var tickinterval = null, insidetick = 0;
function globaltick(rps, fn) {
  if (fn === undefined && $.isFunction(rps)) {
    fn = rps;
    rps = 1;
  }
  if (tickinterval) {
    global.clearInterval(tickinterval);
    tickinterval = null;
  }
  if (fn && rps) {
    tickinterval = global.setInterval(
      function() {
        // Set default speed to Infinity within tick().
        try {
          insidetick++;
          fn();
        } finally {
          insidetick--;
        }
      },
      1000 / rps);
  }
}

// Allow speed to be set in moves per second.
function globaldefaultspeed(mps) {
  if (mps === undefined) {
    return 1000 / $.fx.speeds.turtle;
  } else {
    $.fx.speeds.turtle = mps > 0 ? 1000 / mps : 0;
  }
}

// Simplify $('#x').move() to just x.move()
function turtleids(prefix) {
  if (prefix === undefined) {
    prefix = '';
  }
  $('[id]').each(function(j, item) {
    global[prefix + item.id] = $('#' + item.id);
  });
  attaching_ids = true;
}

// Simplify $(window).click(function(e) { x.moveto(e); } to just
// x.moveto(lastclick).
var eventsaver = null;
function turtleevents(prefix) {
  if (prefix === undefined) {
    prefix = 'last';
  }
  if (eventsaver) {
    $(global).off($.map(eventfn, function(x,k) { return k; }).join(' '),
        eventsaver);
  }
  if (prefix || prefix === '') {
    eventsaver = (function(e) {
      // Keep the old instance if possible.
      var names = [prefix + e.type], j;
      if ((e.originalEvent || e) instanceof MouseEvent) {
        names.push(prefix + 'mouse');
      }
      for (j = 0; j < names.length; ++j) {
        var name = names[j], old = global[name], prop;
        if (old && old.__proto__ === e.__proto__) {
          for (prop in old) { if (old.hasOwnProperty(prop)) delete old[prop]; }
          for (prop in e) { if (e.hasOwnProperty(prop)) old[prop] = e[prop]; }
        } else {
          global[name] = e;
        }
      }
    });
    global[prefix + 'mouse'] = new $.Event();
    for (var k in eventfn) {
      global[prefix + k] = new $.Event();
    }
    $(global).on($.map(eventfn, function(x,k) { return k; }).join(' '),
        eventsaver);
  }
}

// autoScrollAfter will detect if the body is scrolled near the
// bottom already.  If it is, then it autoscrolls it down after
// running the passed function.  (E.g., to allow "print" to scroll
// text upward.)
function autoScrollAfter(f) {
  var slop = 10,
      seen = autoScrollBottomSeen(),
      stick = ($(global).height() + $(global).scrollTop() + slop >=
          $('html').outerHeight(true));
  f();
  if (stick) {
    var scrollPos = $(global).scrollTop(),
        advancedScrollPos = Math.min(seen,
            $('html').outerHeight(true) - $(global).height());
    if (advancedScrollPos > scrollPos) {
      $(global).scrollTop(advancedScrollPos);
    }
  }
}
var autoScrollState = {
  autoScrollTimer: null,
  bottomSeen: 0
};
// We cache bottomSeen until a zero-delay timer can clear it,
// so that a sequence of writes to the screen will not autoscroll
// more than one full page.
function autoScrollBottomSeen() {
  if (!autoScrollState.timer) {
    autoScrollState.timer = setTimeout(function() {
      autoScrollState.timer = null;
    }, 0);
    var offset = $('body').offset();
    var doctop = offset ? offset.top : 8;
    autoScrollState.bottomSeen = Math.min(
        $(global).height() + $(global).scrollTop(),
        $('body').height() + doctop);
  }
  return autoScrollState.bottomSeen;
}
// undoScrollAfter will return the scroll position back to its original
// location after running the passed function.  (E.g., to allow focusing
// a control without autoscrolling.)
function undoScrollAfter(f) {
  var scrollPos = $(global).scrollTop();
  f();
  $(global).scrollTop(scrollPos);
}

//////////////////////////////////////////////////////////////////////////
// OUTPUT AND WIDGETS
// functions to create basic HTML widgets for containing written output
// and controls for reading input.
//////////////////////////////////////////////////////////////////////////

// Simplify output of preformatted text inside a <pre>.
function getTrailingPre() {
  var pre = document.body.lastChild;
  if (!pre || pre.tagName != 'PRE') {
    pre = document.createElement('pre');
    document.body.appendChild(pre);
  }
  return pre;
}

function plainTextPrint() {
  var args = arguments;
  autoScrollAfter(function() {
    var pre = getTrailingPre();
    for (var j = 0; j < args.length; j++) {
      pre.appendChild(document.createTextNode(String(args[j])));
    }
  });
}

function plainBoxPrint(clr, text) {
  var elem = $("<div>").css({
    display: 'inline-block',
    verticalAlign: 'top',
    textAlign: 'center',
    height: '1.2em',
    width: '1.2em',
    maxWidth: '1.2em',
    overflow: 'hidden'
  }).appendTo(getTrailingPre()), finish = function() {
    if (clr != null) { elem.css({background: clr}); }
    if (text != null) { elem.text(text); }
  };
  if (!global_turtle) {
    finish();
  } else {
    var turtle = $(global_turtle);
    moveto.call(turtle, null, elem);
    turtle.eq(0).plan(finish);
  }
}

// Put this output on the screen.  Called some time after prepareOutput
// if the turtle is animating.
function doOutput() {
  var early = this;
  autoScrollAfter(function() {
    early.result.appendTo('body');
    if (early.setup) {
      early.setup();
    }
  });
}

// Prepares some output to create, but doesn't put it on the screen yet.
function prepareOutput(html, tag) {
  var prefix = '<' + tag + ' style="display:table">',
      suffix = '</' + tag + '>';
  if (html === undefined || html === null) {
    // Make empty line when no arguments.
    return {result: $(prefix + '<br>' + suffix)};
  } else if (html.jquery || (html instanceof Element && (html = $(html)))) {
    return {result: html};
  } else {
    var wrapped = false, result = null;
    html = '' + html;
    // Try parsing a tag if possible.
    if (/^\s*<.*>\s*$/.test(html)) {
      result = $(html);
    }
    // If it wasn't a single element, then try to wrap it in an element.
    if (result == null || result.length != 1 || result[0].nodeType != 1) {
      result = $(prefix + html + suffix);
    }
    return {result: result};
  }
}

// Creates and displays a one-shot input menu as a set of.
// radio buttons, each with a specified label.
//
// The menu fires a callback event when the user selects
// an item by clicking or by keyboard.  It has been tested
// to have good keyboard accessibility on IE10, FF, and Chrome.
//
// The choices argument can either be:
// (1) an array of choices (each used as both label and outcome).
// (2) a dictionary mapping text labels to outcomes.
//
// The second argument is a callback called once with the outcome
// value when a choice is made.
//
// If the second argument is omitted, it defaults to a function that
// invokes the outcome value if it is a function.  That way, the
// first argument can be a list of functions or a map from
// text labels to functions.
function prepareMenu(choices, fn) {
  var result = $('<form>')
          .css({display:'table',marginLeft:'20px'})
          .submit(function(){return false;}),
      triggered = false,
      count = 0,
      cursor = 0,
      suppressChange = 0,
      keys = {},
      text;
  // Default behavior: invoke the outcome if it is a function.
  if (!fn) {
    fn = (function invokeOutcome(out) {
      if ($.isFunction(out)) { out.call(null); }
    });
  }
  // Creates a function to be called when the user commits and picks
  // a choice: triggering should only be done once.
  function triggerOnce(outcome) {
    return (function(e) {
      if (!triggered && !(suppressChange && e.type == 'change')) {
        triggered = true;
        $(this).prop('checked', true);
        result.find('input[type=radio]').prop('disabled', true);
        fn(outcome);
      }
    });
  }
  // Returns a handler to be called when the user tentatively
  // focuses on the nth item.
  function triggerFocus(ordinal) {
    return (function() {
      if (!triggered) {
        cursor = ordinal;
        focusCursor();
      }
    });
  }
  // Shows keyboard focus rectangle (for browsers that support it)
  // and checks the item under the cursor while suppressing
  // an action.
  function focusCursor(initial) {
    if (!initial) {
      suppressChange += 1;
      setTimeout(function() { suppressChange -= 1; }, 0);
      result.find('input').eq(cursor).prop('checked', true);
    }
    result.find('input').eq(cursor).focus();
  }
  // Constructs the HTML for a choice, with a name that
  // causes the radio buttons to be grouped, and with the
  // text label selected by the programmer.  Also keeps track
  // of the first character of the label for use as a keyboard shortcut.
  function addChoice(text, outcome) {
    if ($.isFunction(text)) {
      // For an array of functions, just label each choice with
      // the ordinal position.
      text = (count + 1).toString();
    }
    var value = $.isFunction(outcome) || outcome == null ? text: outcome,
        radio = $('<input type="radio" name="menu" class="turtleinput">')
            .attr('value', value)
            .on('change click', triggerOnce(outcome)),
        label = $('<label style="display:table">')
            .append(radio).append(text)
            .on('click', triggerOnce(outcome))
            .on('mousedown', triggerFocus(count)),
        key = text && text.toString().substr(0, 1).toUpperCase();
    if (key && !(key in keys)) {
      keys[key] = count;
    }
    count += 1;
    result.append(label);
  }
  // Decodes choices from either an array or a plain object.
  if ($.isArray(choices)) {
    for (var j = 0; j < choices.length; ++j) {
      addChoice(choices[j], choices[j]);
    }
  } else if ($.isPlainObject(choices)) {
    for (text in choices) {
      addChoice(text, choices[text]);
    }
  } else {
    addChoice(choices, choices);
  }
  // Accessibility support: deal with arrow keys.
  result.on('keydown', function(e) {
    if (e.which >= 37 && e.which <= 40 || e.which == 32) {
      var synccursor = result.find('input').index(result.find(':checked'));
      if (synccursor >= 0 && cursor != synccursor) {
        // If the highlighted item was moved in a way that we didn't
        // track, then just let it show what the browser switched to.
        cursor = synccursor;
      } else {
        if (synccursor < 0) { // If unselected, first arrow selects the first.
          cursor = 0;
        } else if (e.which >= 39 || e.which == 32) { // Cycle forward.
          cursor = (cursor + 1) % count;
        } else { // Cycle backward.
          cursor = (cursor + count - 1) % count;
        }
      }
      focusCursor();
      return false;  // Suppress browser's default handling.
    } else if (e.which == 13) {
      // Enter key will proceed.
      result.find(':checked').click();
    } else if (String.fromCharCode(e.which) in keys) {
      cursor = keys[String.fromCharCode(e.which)];
      focusCursor();
    }
  });
  return {
    result: result,
    setup: function() {
      // Focus, but don't cause autoscroll to occur due to focus.
      undoScrollAfter(function() { focusCursor(true); });
    }
  }
}

// Simplify $('body'>.append('<button>' + label + '</button>').click(fn).
function prepareButton(name, callback) {
  if ($.isFunction(name) && callback === undefined) {
    callback = name;
    name = null;
  }
  if (name === null || name === undefined) {
    name = 'button';
  }
  var result = $('<button class="turtleinput">' +
      escapeHtml(name) + '</button>');
  if (callback) {
    result.click(callback);
  }
  return {result: result};
}

//////////////////////////////////////////////////////////////////////////
// ONE-SHOT INPUT SUPPORT
// for creating an input box with a label, for one-shot input
//////////////////////////////////////////////////////////////////////////

var microphoneSvg = "data:image/svg+xml,<svg%20xmlns=%22http://www.w3.org/2000/svg%22%20viewBox=%220%200%20260%20400%22><path%20d=%22M180,210c0,26-22,48-48,48h-12c-26,0-48-22-48-48v-138c0-26,22-48,48-48h12c26,0,48,22,48,48zm51,-31h-9c-5,0-9,4-9,9v8c0,50-37,91-87,91c-50,0-87-41-87-91v-8c0-5-4-9-9-9h-9c-5,0-9,4-9,9v8c0,59,40,107,96,116v37h-34c-5,0-9,4-9,9v18c0,5,4,9,9,9h105c5,0,9-4,9-9v-18c0-5-5-9-9-9h-34v-37c56-9,96-58,96-116v-8c0-5-4-9-9-9%22/></svg>"

// Simplify $('body').append('<input>' + label) and onchange hookup.
// Type can be 'auto', 'number', 'text', or 'voice' for slightly
// different interfaces.
function prepareInput(name, callback, type) {
  if ($.isFunction(name) && !callback) {
    callback = name;
    name = null;
  }
  if (!type) { type = 'auto'; }
  name = $.isNumeric(name) || name ? name : '&rArr;';
  var textbox = $('<input class="turtleinput">').css({margin:0, padding:0}),
      button = $('<button>Submit</button>').css({marginLeft:4}),
      label = $('<label style="display:table">' + name + '&nbsp;' +
        '</label>').append(textbox).append(button),
      debounce = null,
      lastseen = textbox.val(),
      recognition = null;
  function dodebounce() {
    if (!debounce) {
      debounce = setTimeout(function() { debounce = null; }, 1000);
    }
  }
  function newval() {
    if (!validate()) { return false; }
    var val = textbox.val();
    if (debounce && lastseen == val) { return; }
    dodebounce();
    lastseen = val;
    textbox.remove();
    button.remove();
    label.append(val).css({display: 'table'});
    if (type == 'number' || (type == 'auto' &&
      $.isNumeric(val) && ('' + parseFloat(val) == val))) {
      val = parseFloat(val);
    }
    label.prop('value', val);
    if (callback) { setTimeout(function() {callback.call(label, val); }, 0); }
  }
  function validate() {
    if (type != 'numeric') return true;
    var val = textbox.val(),
        nval = val.replace(/[^0-9\.]/g, '');
    if (val != nval || !$.isNumeric(nval)) {
      textbox.val(nval);
      return false;
    }
    return true;
  }
  function key(e) {
    if (e.which == 13) {
      if (!validate()) { return false; }
      newval();
    }
    if (type == 'voice' && recognition) {
      recognition.abort();
      recognition = null;
    }
    if (type == 'numeric' && (e.which >= 32 && e.which <= 127) &&
        (e.which < '0'.charCodeAt(0) || e.which > '9'.charCodeAt(0)) &&
        (e.which != '.'.charCodeAt(0) || ~textbox.val().indexOf('.'))) {
      return false;
    }
  }
  textbox.on('keypress keydown', key);
  button.on('click', newval);
  return {
    result: label,
    setup: function() {
      dodebounce();
      if (type == 'text' || type == 'voice') {
        // Widen a "readstr" textbox to make it fill the line.
        var availwidth = label.parent().width(),
            freewidth = availwidth + label.offset().left - textbox.offset().left,
            bigwidth = Math.max(256, availwidth / 2),
            desiredwidth = freewidth < bigwidth ? availwidth : freewidth,
            marginwidth = textbox.outerWidth(true) - textbox.width();
        textbox.width(desiredwidth - marginwidth);
      }
      if (type == 'number') {
        textbox.attr('type', 'number');
      }
      if (type == 'voice') {
        button.css({display: 'none'});
        var SR = global.SpeechRecognition || global.webkitSpeechRecognition;
        if ('function' == typeof(SR)) {
          try {
            recognition = new SR();
            recognition.continuous = false;
            recognition.interimResults = true;
            textbox.css({backgroundColor: 'lightyellow',
              color: 'gray',
              backgroundImage: "url(" + microphoneSvg + ")",
              backgroundRepeat: 'no-repeat',
              backgroundPosition: 'center'});
            recognition.onspeechstart = function() {
              textbox.css({background: 'lightgreen'});
            };
            recognition.onend = function() {
              textbox.css({color: '', backgroundColor: '', backgroundImage: '',
                backgroundRepeat: '', backgroundPosition: ''});
              textbox.val(lastseen);
              newval();
            };
            recognition.onresult = function(event) {
              var text = event.results[0][0].transcript;
              var confidence = event.results[0][0].confidence;
              var shade = 128 - 128 * confidence;
              if (event.results[0].isFinal) {
                shade = 0;
                lastseen = text;
              }
              textbox.css({color: componentColor('rgb', shade, shade, shade)});
              textbox.val(text);
            };
            recognition.start();
          } catch (e) {
            console.log(e);
          }
        }
      }
      // Focus, but don't cause autoscroll to occur due to focus.
      undoScrollAfter(function() { textbox.focus(); });
    }
  };
}

//////////////////////////////////////////////////////////////////////////
// IMAGE PRINTER
//////////////////////////////////////////////////////////////////////////

// Simplify creation of images.
function prepareImage(url, options) {
  if ($.isNumeric(options)) {
    options = { height: options };
  }
  var result = $('<img>');
  if (url) {
    result.attr('src', imgUrl(url));
  }
  if (options) {
    result.css(options);
  }
  return {
    result: result
  };
}

//////////////////////////////////////////////////////////////////////////
// TABLE PRINTER
//////////////////////////////////////////////////////////////////////////

// Simplify creation of tables with cells.
function prepareTable(height, width, cellCss, tableCss) {
  var contents = null, row, col;
  if ($.isArray(height)) {
    tableCss = cellCss;
    cellCss = width;
    contents = height;
    height = contents.length;
    width = 0;
    for (row = 0; row < height; row++) {
      if ($.isArray(contents[row])) {
        width = Math.max(width, contents[row].length);
      } else {
        width = Math.max(width, 1);
      }
    }
  }
  var html = ['<table>'];
  for (row = 0; row < height; row++) {
    html.push('<tr>');
    for (col = 0; col < width; col++) {
      if (contents) {
        if ($.isArray(contents[row]) && col < contents[row].length) {
          html.push('<td>' + escapeHtml(contents[row][col]) + '</td>');
        } else if (!$.isArray(contents[row]) && col == 0) {
          html.push('<td>' + escapeHtml(contents[row]) + '</td>');
        } else {
          html.push('<td></td>');
        }
      } else {
        html.push('<td></td>');
      }
    }
    html.push('</tr>');
  }
  html.push('</table>');
  var result = $(html.join(''));
  var defaultCss = {
    borderCollapse: 'collapse',
    width: '35px',
    height: '35px',
    border: '1px solid black',
    tableLayout: 'fixed',
    textAlign: 'center',
    margin: '0',
    padding: '0'
  };
  result.css($.extend({}, defaultCss,
    { width: 'auto', height: 'auto', maxWidth: 'auto', border: 'none'},
    tableCss));
  result.find('td').css($.extend({}, defaultCss, cellCss));
  return {
    result: result
  };
}


//////////////////////////////////////////////////////////////////////////
// COLOR SUPPORT
// TODO: import chroma.js
//////////////////////////////////////////////////////////////////////////
// Functions to generate CSS color strings
function componentColor(t, args) {
  return t + '(' + Array.prototype.join.call(args, ',') + ')';
}

function autoArgs(args, start, map) {
  var j = 0;
  var taken = [];
  var result = {};
  for (var key in map) {
    var pattern = map[key];
    for (j = start; j < args.length; ++j) {
      if (~taken.indexOf(j)) continue;
      if (pattern == '*') {
        break;
      } else if (pattern instanceof RegExp && pattern.test(args[j])) {
        break;
      } else if (pattern instanceof Function && pattern(args[j])) {
        break;
      } else if (pattern == typeof args[j]) {
        break;
      }
    }
    if (j < args.length) {
      taken.push(j);
      result[key] = args[j];
    }
  }
  if (taken.length + start < args.length) {
    var extra = [];
    for (j = start; j < args.length; ++j) {
      if (~taken.indexOf(j)) continue;
      extra.push(args[j]);
    }
    result.extra = extra;
  }
  return result;
}


//////////////////////////////////////////////////////////////////////////
// DEBUGGING SUPPORT
//////////////////////////////////////////////////////////////////////////
var debug = {
  init: function initdebug() {
    if (this.ide) return;  // Don't re-initialize debug.
    try {
      if (parent && parent.ide && parent.ide.bindframe &&
          parent.ide.bindframe(global, parent)) {
        this.ide = parent.ide;
        this.attached = true;
      }
    } catch(e) { }
    if (this.attached) {
      if (global.addEventListener) {
        global.addEventListener('error', function(event) {
          // An error event will highlight the error line.
          debug.reportEvent('error', [event]);
        });
      }
    }
  },
  attached: false,
  ide: null,
  reportEvent: function reportEvent(name, args) {
    if (this.ide) { this.ide.reportEvent(name, args); }
  },
  nextId: function nextId() {
    if (this.ide) {
      return this.ide.nextId();
    } else {
      return 0;
    }
  }
};

debug.init();

//////////////////////////////////////////////////////////////////////////
// X Y coordinate showing support
//////////////////////////////////////////////////////////////////////////
(function() {
  if (!debug.ide) {
    // Only show the X-Y tip if inside a debugging IDE.
    return;
  }
  var location = $('<samp>').css({
    position: 'fixed',
    zIndex: 1e6-1,
    fontFamily: 'sans-serif',
    display: 'none',
    background: '#ff8',
    border: '1px solid dimgray',
    padding: '1px',
    cursor: 'move',
    fontSize: 12
  }).appendTo('body');
  function tr(n) {
    return n.toFixed(1).replace(/\.0$/, '');
  }
  function cd(s) {
    return '<code style="font-weight:bold;color:blue">' + s + '</code>';
  }
  function cont(loc, ev) {
    return loc && (loc = loc.get(0)) && ev && ev.target &&
           (loc == ev.target || $.contains(loc, ev.target));
  }
  var linestart = null, linecanvas = null, lineend = null,
      xa = 0, ya = 0, xb = 0, yb = 0, xt, yt, dr, ar;
  $(global).on('mousedown mouseup mousemove keydown', function(e) {
    if (e.type == 'keydown') {
      if (e.which < 27) return;
      if (linecanvas) linecanvas.remove();
      linecanvas = linestart = lineend = null;
    }
    if (e.type == 'mousedown') {
      if (!linecanvas) {
        if (cont(location, e)) {
          // State 1: click on the tooltip to move it.
          var w = sizexy();
          lineend = linestart = null;
          linecanvas = $('<canvas width="' + w[0] + '" height="' + w[1] + '">').
             css({
            position: 'absolute',
            top: 0,
            left: 0,
            cursor: 'crosshair',
            zIndex: 1e6
          }).appendTo('body');
        }
      } else if (lineend) {
        // State 4: Click to remove everything.
        if (linecanvas) { linecanvas.remove(); }
        linecanvas = linestart = lineend = null;
      } else if (linestart) {
        // State 3: Click to pin the line to the end.
        linecanvas.css({cursor: 'default'});
        lineend = e;
      } else {
        // State 1: Click to plant the line start.
        $.turtle.interrupt('reset');
        var touched = $('.turtle').within('touch', e);
        linestart = touched.length ? touched.eq(0) : e;
      }
    }
    if (linecanvas) {
      var cnv = linecanvas.canvas(),
          c = cnv.getContext('2d'),
          relative = false,
          p = lineend || e,
          html, dx, dy, dd, dir, ang;
      if (linestart && 'function' == typeof(linestart.pagexy)) {
        var xy = linestart.getxy(), s = linestart.pagexy();
        s.x = xy[0];
        s.y = xy[1];
        relative = true;
        dir = linestart.direction();
        html = [
          'getxy is ' + tr(s.x) + ', ' + tr(s.y),
          'direction is ' + tr(dir)
        ];
      } else {
        s = linestart || p;
        html = [
          cd('moveto ' + tr(s.x) + ', ' + tr(s.y))
        ];
      }
      html.unshift(lineend ?
        '<span style="color:green">click to close</span>' :
        linestart ?
        '<span style="color:red">click to measure</span>' :
        '<span style="color:red">click on point</span>'
      );
      dx = p.x - s.x,
      dy = p.y - s.y,
      dd = Math.sqrt(dx * dx + dy * dy),
      ang = Math.atan2(dx, dy) / Math.PI * 180;
      if (linestart) {
        c.save();
        c.clearRect(xa - 10, ya - 10, xb - xa + 20, yb - ya + 20);
        xa = xb = s.pageX;
        ya = yb = s.pageY;
        // Draw a dot
        c.fillStyle = 'red';
        c.beginPath();
        c.arc(s.pageX, s.pageY, 4, 0, 2*Math.PI, false);
        c.closePath();
        c.fill();
        if (dd > 0) {
          if (relative) {
            c.strokeStyle = 'black';
            c.fillStyle = 'black';
            dr = (dir - 90) / 180 * Math.PI;
            ar = (ang - 90) / 180 * Math.PI;
            xt = s.pageX + Math.cos(dr) * 100;
            yt = s.pageY + Math.sin(dr) * 100;
            drawArrowLine(c, 2, s.pageX, s.pageY, xt, yt);
            xa = Math.min(xt, xa);
            ya = Math.min(yt, ya);
            xb = Math.max(xt, xb);
            yb = Math.max(yt, yb);
            var delta = (360 + ang - dir) % 360;
            c.beginPath();
            if (delta <= 180) {
              html.push(cd('rt ' + tr(delta)));
              if (dd >= 20) c.arc(s.pageX, s.pageY, 20, dr, ar);
            } else {
              html.push(cd('lt ' + tr(360 - delta)));
              if (dd >= 20) c.arc(s.pageX, s.pageY, 20, ar, dr);
            }
            c.stroke();
            xa = Math.min(s.pageX - 20, xa)
            ya = Math.min(s.pageY - 20, ya)
            xb = Math.max(s.pageX + 20, xb)
            yb = Math.max(s.pageY + 20, yb)
          } else {
            html.push(cd('turnto ' + tr(ang)));
          }
          html.push(cd('fd ' + tr(dd)));
          html.push('end at ' + tr(p.x) + ', ' + tr(p.y));
          // Draw an arrow.
          c.strokeStyle = 'red';
          c.fillStyle = 'red';
          drawArrowLine(c, 2, s.pageX, s.pageY, p.pageX, p.pageY);
          xa = Math.min(p.pageX, xa);
          ya = Math.min(p.pageY, ya);
          xb = Math.max(p.pageX, xb);
          yb = Math.max(p.pageY, yb);
        }
        c.restore();
      }
      location.css({left:0, top:0}).html(html.join('<br>')).show();
      // Position the draggable tip to the side away from the start point.
      var pos = { left: '', top: '', right: '', bottom: '' };
      if (p.pageX + 5 < s.pageX) {
        pos.left = Math.max(
            p.pageX - $(global).scrollLeft() - location.outerWidth() - 5, 2);
      } else {
        pos.left = Math.min(p.pageX - $(global).scrollLeft() + 5,
            $(document).width() - location.outerWidth() - 2);
      }
      if (p.pageY + 5 < s.pageY) {
        pos.top = Math.max(
            p.pageY - $(global).scrollTop() - location.outerHeight() - 5, 2);
      } else {
        pos.top = Math.min(p.pageY - $(global).scrollTop() + 5,
            $(document).height() - location.outerHeight() - 2);
      }
      location.css(pos);
    } else {
      html = [];
      if (cont(location, e)) {
        html.push('<span style="color:red">click to use</span>');
      }
      if (e.x != null) {
        html.push(e.x + ', ' + e.y);
      }
      location.html(html.join('<br>')).css({
        left: '',
        top: '',
        right: 0,
        bottom: 0
      }).show();
    }
  });
})();


//////////////////////////////////////////////////////////////////////////
// SEE LOGGING SUPPORT
// A copy of see.js here.
// TODO: figure out how to move this into the IDE.
//////////////////////////////////////////////////////////////////////////

// see.js version 0.2

var pulljQueryVersion = null;  // Disable auto-pull of jQuery

var seepkg = 'see'; // Defines the global package name used.
var version = '0.2';
var oldvalue = noteoldvalue(seepkg);
// Option defaults
var linestyle = 'position:relative;display:block;font-family:monospace;' +
  'font-size:16px;word-break:break-all;margin-bottom:3px;padding-left:1em;';
var logdepth = 5;
var autoscroll = false;
var logelement = 'body';
var panel = 'auto';
try {
  // show panel by default if framed inside a an ide,
  // and if the screen is big enough (i.e., omit mobile clients).
  if (global.self !== global.top &&
      screen.width >= 800 && screen.height >= 600 &&
      parent && parent.ide) { panel = parent.ide.getOptions().panel; }
} catch(e) {}
var consolelog = panel;
var see;  // defined below.
var paneltitle = '';
var logconsole = null;
var uselocalstorage = '_loghistory';
var panelheight = 50;
var currentscope = '';
var scopes = {
  '':  { e: global.eval, t: global },
  top: { e: global.eval, t: global }
};
var coffeescript = global.CoffeeScript;
var seejs = '(function(){return eval(arguments[0]);})';

function init(options) {
  if (arguments.length === 0) {
    options = {};
  } else if (arguments.length == 2) {
    var newopt = {};
    newopt[arguments[0]] = arguments[1];
    options = newopt;
  } else if (arguments.length == 1 && typeof arguments[0] == 'function') {
    options = {'eval': arguments[0]};
  }
  if ('jQuery' in options) { $ = options.jQuery; }
  if ('eval' in options) { scopes[''].e = options['eval']; }
  if ('this' in options) { scopes[''].t = options['this']; }
  if ('element' in options) { logelement = options.element; }
  if ('autoscroll' in options) { autoscroll = options.autoscroll; }
  if ('linestyle' in options) { linestyle = options.linestyle; }
  if ('depth' in options) { logdepth = options.depth; }
  if ('panel' in options) { panel = options.panel; }
  if ('height' in options) { panelheight = options.height; }
  if ('title' in options) { paneltitle = options.title; }
  if ('console' in options) { logconsole = options.console; }
  if ('history' in options) { uselocalstorage = options.history; }
  if ('coffee' in options) { coffeescript = options.coffee; }
  if ('abbreviate' in options) { abbreviate = options.abbreviate; }
  if ('consolehook' in options) { consolehook = options.consolehook; }
  if ('consolelog' in options) { consolelog = options.consolelog; }
  if ('noconflict' in options) { noconflict(options.noconflict); }
  if (panel) {
    // panel overrides element and autoscroll.
    logelement = '#_testlog';
    autoscroll = '#_testscroll';
    if (panel === true) {
      startinitpanel();
    }
  }
  if (consolelog === true) { initconsolelog(); }
  return scope();
}

function scope(name, evalfuncarg, evalthisarg) {
  if (arguments.length <= 1) {
    if (!arguments.length) {
      name = '';
    }
    return seepkg + '.scope(' + cstring(name) + ',' + seejs + ',this)';
  }
  scopes[name] = { e: evalfuncarg, t: evalthisarg };
}

function seeeval(scope, code) {
  if (arguments.length == 1) {
    code = scope;
    scope = '';
  }
  var ef = scopes[''].e, et = scopes[''].t;
  if (scopes.hasOwnProperty(scope)) {
    if (scopes[scope].e) { ef = scopes[scope].e; }
    if (scopes[scope].t) { et = scopes[scope].t; }
  }
  debug.reportEvent("seeeval", [scope, code]);
  return ef.call(et, code);
}

var varpat = '[_$a-zA-Z\xA0-\uFFFF][_$a-zA-Z0-9\xA0-\uFFFF]*';
var initialvardecl = new RegExp(
  '^\\s*var\\s+(?:' + varpat + '\\s*,\\s*)*' + varpat + '\\s*;\\s*');

function barecs(s) {
  // Compile coffeescript in bare mode.
  var compiler = coffeescript || global.CoffeeScript;
  var compiled = compiler.compile(s, {bare:1});
  if (compiled) {
    // Further strip top-level var decls out of the coffeescript so
    // that assignments can leak out into the enclosing scope.
    compiled = compiled.replace(initialvardecl, '');
  }
  return compiled;
}

function exportsee() {
  see.repr = repr;
  see.html = loghtml;
  see.noconflict = noconflict;
  see.init = init;
  see.scope = scope;
  see.eval = seeeval;
  see.barecs = barecs;
  see.here = 'eval(' + seepkg + '.init())';
  see.clear = seeclear;
  see.hide = seehide;
  see.show = seeshow;
  see.visible = seevisible;
  see.enter = seeenter;
  see.js = seejs;
  see.cs = '(function(){return eval(' + seepkg + '.barecs(arguments[0]));})';
  see.version = version;
  global[seepkg] = see;
}

function noteoldvalue(name) {
  return {
    name: name,
    has: global.hasOwnProperty(name),
    value: global[name]
  };
}

function restoreoldvalue(old) {
  if (!old.has) {
    delete global[old.name];
  } else {
    global[old.name] = old.value;
  }
}

function noconflict(newname) {
  if (!newname || typeof(newname) != 'string') {
    newname = 'see' + (1 + Math.random() + '').substr(2);
  }
  if (oldvalue) {
    restoreoldvalue(oldvalue);
  }
  seepkg = newname;
  oldvalue = noteoldvalue(newname);
  exportsee();
  return see;
}

function pulljQuery(callback) {
  if (!pulljQueryVersion || ($ && $.fn && $.fn.jquery)) {
    callback();
    return;
  }
  function loadscript(src, callback) {
    function setonload(script, fn) {
      script.onload = script.onreadystatechange = fn;
    }
    var script = document.createElement("script"),
       head = document.getElementsByTagName("head")[0],
       pending = 1;
    setonload(script, function() {
      if (pending && (!script.readyState ||
          {loaded:1,complete:1}[script.readyState])) {
        pending = 0;
        callback();
        setonload(script, null);
        head.removeChild(script);
      }
    });
    script.src = src;
    head.appendChild(script);
  }
  loadscript(
      '//ajax.googleapis.com/ajax/libs/jquery/' +
      pulljQueryVersion + '/jquery.min.js',
      function() {
    $ = jQuery.noConflict(true);
    callback();
  });
}

// ---------------------------------------------------------------------
// LOG FUNCTION SUPPORT
// ---------------------------------------------------------------------
var logcss = "input._log:focus{outline:none;}samp._logcaret{position:absolute;left:0;font-size:120%;}samp._logcaret:before{content: '>'}label._log > span:first-of-type:hover{text-decoration:underline;}samp._log > label._log,samp_.log > span > label._log{display:inline-block;vertical-align:top;}label._log > span:first-of-type{margin-left:2em;text-indent:-1em;}label._log > ul{display:none;padding-left:14px;margin:0;}label._log > span:before{content:'';font-size:70%;font-style:normal;display:inline-block;width:0;text-align:center;}label._log > span:first-of-type:before{content:'\\0025B6';}label._log > ul > li{display:block;white-space:pre-line;margin-left:2em;text-indent:-1em}label._log > ul > li > samp{margin-left:-1em;text-indent:0;white-space:pre;}label._log > input[type=checkbox]:checked ~ span{margin-left:2em;text-indent:-1em;}label._log > input[type=checkbox]:checked ~ span:first-of-type:before{content:'\\0025BC';}label._log > input[type=checkbox]:checked ~ span:before{content:'';}label._log,label._log > input[type=checkbox]:checked ~ ul{display:block;}label._log > span:first-of-type,label._log > input[type=checkbox]:checked ~ span{display:inline-block;}label._log > input[type=checkbox],label._log > input[type=checkbox]:checked ~ span > span{display:none;}";
var addedcss = false;
var cescapes = {
  '\0': '\\0', '\b': '\\b', '\f': '\\f', '\n': '\\n', '\r': '\\r',
  '\t': '\\t', '\v': '\\v', "'": "\\'", '"': '\\"', '\\': '\\\\'
};
var retrying = null;
var queue = [];

see = function see() {
  if (logconsole && typeof(logconsole.log) == 'function') {
    logconsole.log.apply(global.console, arguments);
  }
  var args = Array.prototype.slice.call(arguments);
  queue.push('<samp class="_log">');
  while (args.length) {
    var obj = args.shift();
    if (vtype(obj) == 'String')  {
      // Logging a string just outputs the string without quotes.
      queue.push(htmlescape(obj));
    } else {
      queue.push(repr(obj, logdepth, queue));
    }
    if (args.length) { queue.push(' '); }
  }
  queue.push('</samp>');
  flushqueue();
};

function loghtml(html) {
  queue.push('<samp class="_log">');
  queue.push(html);
  queue.push('</samp>');
  flushqueue();
}

function vtype(obj) {
  var bracketed = Object.prototype.toString.call(obj);
  var vt = bracketed.substring(8, bracketed.length - 1);
  if (vt == 'Object') {
    if ('length' in obj && 'slice' in obj && 'number' == typeof obj.length) {
      return 'Array';
    }
    if ('originalEvent' in obj && 'target' in obj && 'type' in obj) {
      return vtype(obj.originalEvent);
    }
  }
  return vt;
}

function isprimitive(vt) {
  switch (vt) {
    case 'String':
    case 'Number':
    case 'Boolean':
    case 'Undefined':
    case 'Date':
    case 'RegExp':
    case 'Null':
      return true;
  }
  return false;
}

function isdom(obj) {
  return (obj && obj.nodeType && obj.nodeName &&
          typeof(obj.cloneNode) == 'function');
}

function midtruncate(s, maxlen) {
  if (maxlen && maxlen > 3 && s.length > maxlen) {
    return s.substring(0, Math.floor(maxlen / 2) - 1) + '...' +
        s.substring(s.length - (Math.ceil(maxlen / 2) - 2));
  }
  return s;
}

function cstring(s, maxlen) {
  function cescape(c) {
    if (cescapes.hasOwnProperty(c)) {
      return cescapes[c];
    }
    var temp = '0' + c.charCodeAt(0).toString(16);
    return '\\x' + temp.substring(temp.length - 2);
  }
  if (s.indexOf('"') == -1 || s.indexOf('\'') != -1) {
    return midtruncate('"' +
        htmlescape(s.replace(/[\0-\x1f\x7f-\x9f"\\]/g, cescape)) + '"', maxlen);
  } else {
    return midtruncate("'" +
        htmlescape(s.replace(/[\0-\x1f\x7f-\x9f'\\]/g, cescape)) + "'", maxlen);
  }
}
function tiny(obj, maxlen) {
  var vt = vtype(obj);
  if (vt == 'String') { return cstring(obj, maxlen); }
  if (vt == 'Undefined' || vt == 'Null') { return vt.toLowerCase(); }
  if (isprimitive(vt)) { return '' + obj; }
  if (vt == 'Array' && obj.length === 0) { return '[]'; }
  if (vt == 'Object' && isshort(obj)) { return '{}'; }
  if (isdom(obj) && obj.nodeType == 1) {
    if (obj.hasAttribute('id')) {
      return obj.tagName.toLowerCase() +
          '#' + htmlescape(obj.getAttribute('id'));
    } else {
      if (obj.hasAttribute('class')) {
        var classname = obj.getAttribute('class').split(' ')[0];
        if (classname) {
          return obj.tagName.toLowerCase() + '.' + htmlescape(classname);
        }
      }
      return obj.tagName.toLowerCase();
    }
  }
  return vt;
}
function isnonspace(dom) {
  return (dom.nodeType != 3 || /[^\s]/.exec(dom.textContent));
}
function trimemptystartline(s) {
  return s.replace(/^\s*\n/, '');
}
function isshort(obj, shallow, maxlen) {
  var vt = vtype(obj);
  if (isprimitive(vt)) { return true; }
  if (!shallow && vt == 'Array') { return !maxlen || obj.length <= maxlen; }
  if (isdom(obj)) {
    if (obj.nodeType == 9 || obj.nodeType == 11) return false;
    if (obj.nodeType == 1) {
      return (obj.firstChild === null ||
         obj.firstChild.nextSibling === null &&
         obj.firstChild.nodeType == 3 &&
         obj.firstChild.textContent.length <= maxlen);
    }
    return true;
  }
  if (vt == 'Function') {
    var sc = obj.toString();
    return (sc.length - sc.indexOf('{') <= maxlen);
  }
  if (vt == 'Error') {
    return !!obj.stack;
  }
  var count = 0;
  for (var prop in obj) {
    if (obj.hasOwnProperty(prop)) {
      count += 1;
      if (shallow && !isprimitive(vtype(obj[prop]))) { return false; }
      if (maxlen && count > maxlen) { return false; }
    }
  }
  return true;
}
function domsummary(dom, maxlen) {
  var short;
  if ('outerHTML' in dom) {
    short = isshort(dom, true, maxlen);
    var html = dom.cloneNode(short).outerHTML;
    var tail = null;
    if (!short) {
      var m = /^(.*)(<\/[^\s]*>$)/.exec(html);
      if (m) {
        tail = m[2];
        html = m[1];
      }
    }
    return [htmlescape(html), tail && htmlescape(tail)];
  }
  if (dom.nodeType == 1) {
    var parts = ['<' + dom.tagName];
    for (var j = 0; j < dom.attributes.length; ++j) {
      parts.push(domsummary(dom.attributes[j], maxlen)[0]);
    }
    short = isshort(dom, true, maxlen);
    if (short && dom.firstChild) {
      return [htmlescape(parts.join(' ') + '>' +
          dom.firstChild.textContent + '</' + dom.tagName + '>'), null];
    }
    return [htmlescape(parts.join(' ') + (dom.firstChild? '>' : '/>')),
        !dom.firstChild ? null : htmlescape('</' + dom.tagName + '>')];
  }
  if (dom.nodeType == 2) {
    return [htmlescape(dom.name + '="' +
        htmlescape(midtruncate(dom.value, maxlen), '"') + '"'), null];
  }
  if (dom.nodeType == 3) {
    return [htmlescape(trimemptystartline(dom.textContent)), null];
  }
  if (dom.nodeType == 4) {
    return ['<![CDATA[' + htmlescape(midtruncate(dom.textContent, maxlen)) +
        ']]>', null];
  }
  if (dom.nodeType == 8) {
    return ['<!--' + htmlescape(midtruncate(dom.textContent, maxlen)) +
        '-->', null];
  }
  if (dom.nodeType == 10) {
    return ['<!DOCTYPE ' + htmlescape(dom.nodeName) + '>', null];
  }
  return [dom.nodeName, null];
}
function summary(obj, maxlen) {
  var vt = vtype(obj);
  if (isprimitive(vt)) {
    return tiny(obj, maxlen);
  }
  if (isdom(obj)) {
    var ds = domsummary(obj, maxlen);
    return ds[0] + (ds[1] ? '...' + ds[1] : '');
  }
  if (vt == 'Function') {
    var ft = obj.toString();
    if (ft.length - ft.indexOf('{') > maxlen) {
      ft = ft.replace(/\{(?:.|\n)*$/, '').trim();
    }
    return ft;
  }
  if ((vt == 'Error' || vt == 'ErrorEvent') && 'message' in obj) {
    return obj.message;
  }
  var pieces = [];
  if (vt == 'Array' && obj.length < maxlen) {
    var identical = (obj.length >= 5);
    var firstobj = identical && obj[0];
    for (var j = 0; j < obj.length; ++j) {
      if (identical && obj[j] !== firstobj) { identical = false; }
      pieces.push(tiny(obj[j], maxlen));
    }
    if (identical) {
      return '[' + tiny(firstobj, maxlen) + '] \xd7 ' + obj.length;
    }
    return '[' + pieces.join(', ') + ']';
  } else if (isshort(obj, false, maxlen)) {
    for (var key in obj) {
      if (obj.hasOwnProperty(key)) {
        pieces.push(quotekey(key) + ': ' + tiny(obj[key], maxlen));
      }
    }
    return (vt == 'Object' ? '{' : vt + '{') + pieces.join(', ') + '}';
  }
  if (vt == 'Array') { return 'Array(' + obj.length + ')'; }
  return vt;
}
function quotekey(k) {
  if (/^\w+$/.exec(k)) { return k; }
  return cstring(k);
}
function htmlescape(s, q) {
  var pat = /[<>&]/g;
  if (q) { pat = new RegExp('[<>&' + q + ']', 'g'); }
  return s.replace(pat, function(c) {
    return c == '<' ? '&lt;' : c == '>' ? '&gt;' : c == '&' ? '&amp;' :
           c == '"' ? '&quot;' : '&#' + c.charCodeAt(0) + ';';
  });
}
function unindented(s) {
  s = s.replace(/^\s*\n/, '');
  var leading = s.match(/^\s*\S/mg);
  var spaces = leading.length && leading[0].length - 1;
  var j = 1;
  // If the block begins with a {, ignore those spaces.
  if (leading.length > 1 && leading[0].trim() == '{') {
    spaces = leading[1].length - 1;
    j = 2;
  }
  for (; j < leading.length; ++j) {
    spaces = Math.min(leading[j].length - 1, spaces);
    if (spaces <= 0) { return s; }
  }
  var removal = new RegExp('^\\s{' + spaces + '}', 'mg');
  return s.replace(removal, '');
}
function expand(prefix, obj, depth, output) {
  output.push('<label class="_log"><input type="checkbox"><span>');
  if (prefix) { output.push(prefix); }
  if (isdom(obj)) {
    var ds = domsummary(obj, 10);
    output.push(ds[0]);
    output.push('</span><ul>');
    for (var node = obj.firstChild; node; node = node.nextSibling) {
      if (isnonspace(node)) {
        if (node.nodeType == 3) {
          output.push('<li><samp>');
          output.push(unindented(node.textContent));
          output.push('</samp></li>');
        } else if (isshort(node, true, 20) || depth <= 1) {
          output.push('<li>' + summary(node, 20) + '</li>');
        } else {
          expand('', node, depth - 1, output);
        }
      }
    }
    output.push('</ul>');
    if (ds[1]) {
      output.push('<span>');
      output.push(ds[1]);
      output.push('</span>');
    }
    output.push('</label>');
  } else {
    output.push(summary(obj, 10));
    output.push('</span><ul>');
    var vt = vtype(obj);
    if (vt == 'Function') {
      var ft = obj.toString();
      var m = /\{(?:.|\n)*$/.exec(ft);
      if (m) { ft = m[0]; }
      output.push('<li><samp>');
      output.push(htmlescape(unindented(ft)));
      output.push('</samp></li>');
    } else if (vt == 'Error') {
      output.push('<li><samp>');
      output.push(htmlescape(obj.stack));
      output.push('</samp></li>');
    } else if (vt == 'Array') {
      for (var j = 0; j < Math.min(100, obj.length); ++j) {
        try {
          val = obj[j];
        } catch(e) {
          val = e;
        }
        if (isshort(val, true, 20) || depth <= 1 || vtype(val) == 'global') {
          output.push('<li>' + j + ': ' + summary(val, 100) + '</li>');
        } else {
          expand(j + ': ', val, depth - 1, output);
        }
      }
      if (obj.length > 100) {
        output.push('<li>length=' + obj.length + ' ...</li>');
      }
    } else {
      var count = 0;
      for (var key in obj) {
        if (obj.hasOwnProperty(key)) {
          count += 1;
          if (count > 100) { continue; }
          var val;
          try {
            val = obj[key];
          } catch(e) {
            val = e;
          }
          if (isshort(val, true, 20) || depth <= 1 || vtype(val) == 'global') {
            output.push('<li>');
            output.push(quotekey(key));
            output.push(': ');
            output.push(summary(val, 100));
            output.push('</li>');
          } else {
            expand(quotekey(key) + ': ', val, depth - 1, output);
          }
        }
      }
      if (count > 100) {
        output.push('<li>' + count + ' properties total...</li>');
      }
    }
    output.push('</ul></label>');
  }
}
function initlogcss() {
  if (!addedcss && !global.document.getElementById('_logcss')) {
    var style = global.document.createElement('style');
    style.id = '_logcss';
    style.innerHTML = (linestyle ? 'samp._log{' +
        linestyle + '}' : '') + logcss;
    global.document.head.appendChild(style);
    addedcss = true;
  }
}
function repr(obj, depth, aoutput) {
  depth = depth || 3;
  var output = aoutput || [];
  var vt = vtype(obj);
  if (vt == 'Error' || vt == 'ErrorEvent') {
    output.push('<span style="color:red;">');
    expand('', obj, depth, output);
    output.push('</span>');
  } else if (isprimitive(vt)) {
    output.push(tiny(obj));
  } else if (isshort(obj, true, 100) || depth <= 0) {
    output.push(summary(obj, 100));
  } else {
    expand('', obj, depth, output);
  }
  if (!aoutput) {
    return output.join('');
  }
}
function aselement(s, def) {
  switch (typeof s) {
    case 'string':
      if (s == 'body') { return document.body; }
      if (document.querySelector) { return document.querySelector(s); }
      if ($) { return $(s)[0]; }
      return null;
    case 'undefined':
      return def;
    case 'boolean':
      if (s) { return def; }
      return null;
    default:
      return s;
  }
}
function stickscroll() {
  var stick = false, a = aselement(autoscroll, null);
  if (a) {
    stick = a.scrollHeight - a.scrollTop - 10 <= a.clientHeight;
  }
  if (stick) {
    return (function() {
      a.scrollTop = a.scrollHeight - a.clientHeight;
    });
  } else {
    return (function() {});
  }
}
function flushqueue() {
  var elt = aselement(logelement, null), child;
  if (elt && elt.appendChild && queue.length) {
    initlogcss();
    var temp = global.document.createElement('samp');
    temp.innerHTML = queue.join('');
    queue.length = 0;
    var complete = stickscroll();
    while (child = temp.firstChild) {
      elt.appendChild(child);
    }
    complete();
  }
  if (!retrying && queue.length) {
    if (panel == 'auto') {
      startinitpanel();
    }
    retrying = setTimeout(function() { retrying = null; flushqueue(); }, 100);
  } else if (retrying && !queue.length) {
    clearTimeout(retrying);
    retrying = null;
  }
}

// ---------------------------------------------------------------------
// TEST PANEL SUPPORT
// ---------------------------------------------------------------------
var addedpanel = false;
var initpanelstarted = false;
var inittesttimer = null;
var abbreviate = [{}.undefined];
var consolehook = null;

function seehide() {
  $('#_testpanel').hide();
}
function seeshow() {
  $('#_testpanel').show();
}
function seevisible() {
  return $('#_testpanel').is(':visible');
}
function seeenter(text) {
  $('#_testinput').val(text);
}
function seeclear() {
  if (!addedpanel) { return; }
  $('#_testlog').find('._log').not('#_testpaneltitle').remove();
}
function promptcaret(color) {
  return '<samp class="_logcaret" style="color:' + color + ';"></samp>';
}
function getSelectedText(){
    if(global.getSelection) { return global.getSelection().toString(); }
    else if(document.getSelection) { return document.getSelection(); }
    else if(document.selection) {
        return document.selection.createRange().text; }
}
function formattitle(title) {
  return '<samp class="_log" id="_testpaneltitle" style="font-weight:bold;">' +
      title + '</samp>';
}
var noLocalStorage = null;
function readlocalstorage() {
  if (!uselocalstorage) {
    return;
  }
  var state = { height: panelheight, history: [] }, result;
  try {
    result = global.JSON.parse(global.localStorage[uselocalstorage]);
  } catch(e) {
    result = noLocalStorage || {};
  }
  if (result && result.slice && result.length) {
    // if result is an array, then it's just the history.
    state.history = result;
    return state;
  }
  $.extend(state, result);
  return state;
}
function updatelocalstorage(state) {
  if (!uselocalstorage) {
    return;
  }
  var stored = readlocalstorage(), changed = false;
  if ('history' in state &&
      state.history.length &&
      (!stored.history.length ||
      stored.history[stored.history.length - 1] !==
      state.history[state.history.length - 1])) {
    stored.history.push(state.history[state.history.length - 1]);
    changed = true;
  }
  if ('height' in state && state.height !== stored.height) {
    stored.height = state.height;
    changed = true;
  }
  if (changed) {
    try {
      global.localStorage[uselocalstorage] = global.JSON.stringify(stored);
    } catch(e) {
      noLocalStorage = stored;
    }
  }
}
function wheight() {
  return global.innerHeight || $(global).height();
}
function initconsolelog() {
  try {
    if (consolelog && global.console && !global.console._log &&
        'function' == typeof global.console.log) {
      var _log = global.console._log = global.console.log;
      global.console.log = function log() {
        _log.apply(this, arguments);
        see.apply(this, arguments);
      }
      var _debug = global.console._debug = global.console.debug;
      global.console.debug = function debug() {
        _debug.apply(this, arguments);
        see.apply(this, arguments);
      }
    }
  } catch(e) { }
}
function startinitpanel() {
  if (!initpanelstarted) {
    initpanelstarted = true;
    pulljQuery(tryinitpanel);
  }
}
function tryinitpanel() {
  if (addedpanel) {
    if (paneltitle) {
      if ($('#_testpaneltitle').length) {
        $('#_testpaneltitle').html(paneltitle);
      } else {
        $('#_testlog').prepend(formattitle(paneltitle));
      }
    }
    $('#_testpanel').show();
  } else {
    if (!global.document.getElementById('_testlog') && global.document.body) {
      initconsolelog();
      initlogcss();
      var state = readlocalstorage();
      var titlehtml = (paneltitle ? formattitle(paneltitle) : '');
      if (state.height > wheight() - 50) {
        state.height = Math.min(wheight(), Math.max(10, wheight() - 50));
      }
      $('body').prepend(
        '<samp id="_testpanel" class="turtlefield" ' +
            'style="overflow:hidden;z-index:99;' +
            'position:fixed;bottom:0;left:0;width:100%;height:' + state.height +
            'px;background:rgba(240,240,240,0.8);' +
            'font:10pt monospace;' +
            // This last bit works around this position:fixed bug in webkit:
            // https://code.google.com/p/chromium/issues/detail?id=128375
            '-webkit-transform:translateZ(0);">' +
          '<samp id="_testdrag" style="' +
              'cursor:row-resize;height:6px;width:100%;' +
              'display:block;background:lightgray"></samp>' +
          '<samp id="_testscroll" style="overflow-y:scroll;overflow-x:hidden;' +
             'display:block;width:100%;height:' + (state.height - 6) + 'px;">' +
            '<samp id="_testlog" style="display:block">' +
            titlehtml + '</samp>' +
            '<samp class="_log" style="position:relative;display:block;">' +
            promptcaret('blue') +
            '<input id="_testinput" class="_log" style="width:100%;' +
                'margin:0;border:0;font:inherit;' +
                'background:rgba(255,255,255,0.8);">' +
           '</samp>' +
        '</samp>');
      addedpanel = true;
      flushqueue();
      var historyindex = 0;
      var historyedited = {};
      $('#_testinput').on('keydown', function(e) {
        if (e.which == 13) {
          // Handle the Enter key.
          var text = $(this).val();
          $(this).val('');
          // Save (nonempty, nonrepeated) commands to history and localStorage.
          if (text.trim().length &&
              (!state.history.length ||
               state.history[state.history.length - 1] !== text)) {
            state.history.push(text);
            updatelocalstorage({ history: [text] });
          }
          // Reset up/down history browse state.
          historyedited = {};
          historyindex = 0;
          // Copy the entered prompt into the log, with a grayed caret.
          loghtml('<samp class="_log" style="margin-left:-1em;">' +
                  promptcaret('lightgray') +
                  htmlescape(text) + '</samp>');
          $(this).select();
          // Deal with the ":scope" command
          if (text.trim().length && text.trim()[0] == ':') {
            var scopename = text.trim().substring(1).trim();
            if (!scopename || scopes.hasOwnProperty(scopename)) {
              currentscope = scopename;
              var desc = scopename ? 'scope ' + scopename : 'default scope';
              loghtml('<span style="color:blue">switched to ' + desc + '</span>');
            } else {
              loghtml('<span style="color:red">no scope ' + scopename + '</span>');
            }
            return;
          }
          // When interacting with the see panel, first turn the interrupt
          // flag off to allow commands to operate after an interrupt.
          $.turtle.interrupt('reset');
          // Actually execute the command and log the results (or error).
          var hooked = false;
          try {
            var result;
            try {
              result = seeeval(currentscope, text);
            } finally {
              if (consolehook && consolehook(text, result)) {
                hooked = true;
              } else {
                // Show the result (unless abbreviated).
                for (var j = abbreviate.length - 1; j >= 0; --j) {
                  if (result === abbreviate[j]) break;
                }
                if (j < 0) {
                  loghtml(repr(result));
                }
              }
            }
          } catch (e) {
            // Show errors (unless hooked).
            if (!hooked) {
              see(e);
            }
          }
        } else if (e.which == 38 || e.which == 40) {
          // Handle the up and down arrow keys.
          // Stow away edits in progress (without saving to history).
          historyedited[historyindex] = $(this).val();
          // Advance the history index up or down, pegged at the boundaries.
          historyindex += (e.which == 38 ? 1 : -1);
          historyindex = Math.max(0, Math.min(state.history.length,
              historyindex));
          // Show the remembered command at that slot.
          var newval = historyedited[historyindex] ||
              state.history[state.history.length - historyindex];
          if (typeof newval == 'undefined') { newval = ''; }
          $(this).val(newval);
          this.selectionStart = this.selectionEnd = newval.length;
          e.preventDefault();
        }
      });
      $('#_testdrag').on('mousedown', function(e) {
        var drag = this,
            dragsum = $('#_testpanel').height() + e.pageY,
            barheight = $('#_testdrag').height(),
            dragwhich = e.which,
            dragfunc;
        if (drag.setCapture) { drag.setCapture(true); }
        dragfunc = function dragresize(e) {
          if (e.type != 'blur' && e.which == dragwhich) {
            var winheight = wheight();
            var newheight = Math.max(barheight, Math.min(winheight,
                dragsum - e.pageY));
            var complete = stickscroll();
            $('#_testpanel').height(newheight);
            $('#_testscroll').height(newheight - barheight);
            complete();
          }
          if (e.type == 'mouseup' || e.type == 'blur' ||
              e.type == 'mousemove' && e.which != dragwhich) {
            $(global).off('mousemove mouseup blur', dragfunc);
            if (document.releaseCapture) { document.releaseCapture(); }
            if ($('#_testpanel').height() != state.height) {
              state.height = $('#_testpanel').height();
              updatelocalstorage({ height: state.height });
            }
          }
        };
        $(global).on('mousemove mouseup blur', dragfunc);
        return false;
      });
      $('#_testpanel').on('mouseup', function(e) {
        if (getSelectedText()) { return; }
        // Focus without scrolling.
        var scrollpos = $('#_testscroll').scrollTop();
        $('#_testinput').focus();
        $('#_testscroll').scrollTop(scrollpos);
      });
    }
  }
  if (inittesttimer && addedpanel) {
    clearTimeout(inittesttimer);
  } else if (!addedpanel && !inittesttimer) {
    inittesttimer = setTimeout(tryinitpanel, 100);
  }
}

// Removing this debugging line saves 20kb in minification.
// eval("scope('jquery-turtle', " + seejs + ", this)");

function transparentHull(image, threshold) {
  var c = document.createElement('canvas');
  if (!threshold) threshold = 0;
  c.width = image.width;
  c.height = image.height;
  var ctx = c.getContext('2d');
  ctx.drawImage(image, 0, 0);
  return transparentCanvasHull(c, threshold);
}

function transparentCanvasHull(canvas, threshold) {
  var ctx = canvas.getContext('2d');
  var data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  var hull = [];
  var intthresh = 256 * threshold;
  var first, last, prevfirst = Infinity, prevlast = -1;
  for (var row = 0; row < canvas.height; ++row) {
    // We only take the first/last hull in a row to reduce the number of
    // possible points from O(n^2) to O(n).
    first = Infinity;
    last = -1;
    for (var col = 0; col < canvas.width; ++col) {
      if (data[row * 4 * canvas.width + col * 4 + 3] > intthresh) {
        if (last < 0) first = col;
        last = col;
      }
    }
    if (last >= 0 || prevlast >= 0) {
      hull.push({ pageX: Math.min(first, prevfirst), pageY: row});
      hull.push({ pageX: Math.max(last, prevlast) + 1, pageY: row});
    }
    prevfirst = first;
    prevlast = last;
  }
  if (prevlast >= 0) {
    hull.push({ pageX: prevfirst, pageY: canvas.height});
    hull.push({ pageX: prevlast + 1, pageY: canvas.height});
  }
  return convexHull(hull);
}

function eraseOutsideHull(canvas, hull) {
  var ctx = canvas.getContext('2d'),
      w = canvas.width,
      h = canvas.height,
      j = 0;
  ctx.save();
  // Erase everything outside clipping region.
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.lineTo(w, 0);
  ctx.lineTo(w, h);
  ctx.lineTo(0, h);
  ctx.closePath();
  if (hull.length) {
    ctx.moveTo(hull[0].pageX, hull[0].pageY);
    for (; j < hull.length; j += 1) {
      ctx.lineTo(hull[j].pageX, hull[j].pageY);
    }
  }
  ctx.closePath();
  ctx.clip();
  ctx.clearRect(0, 0, w, h);
  ctx.restore();
}

function scalePolygon(poly, sx, sy, tx, ty) {
  for (var i = 0; i < poly.length; i++){
    poly[i].pageX = poly[i].pageX * sx + tx;
    poly[i].pageY = poly[i].pageY * sy + ty;
  }
}

}).call(this, this.jQuery);
