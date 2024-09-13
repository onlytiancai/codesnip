// Generated from Hello.g4 by ANTLR 4.13.1
import org.antlr.v4.runtime.Lexer;
import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.Token;
import org.antlr.v4.runtime.TokenStream;
import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.atn.*;
import org.antlr.v4.runtime.dfa.DFA;
import org.antlr.v4.runtime.misc.*;

@SuppressWarnings({"all", "warnings", "unchecked", "unused", "cast", "CheckReturnValue", "this-escape"})
public class Hello extends Lexer {
	static { RuntimeMetaData.checkVersion("4.13.1", RuntimeMetaData.VERSION); }

	protected static final DFA[] _decisionToDFA;
	protected static final PredictionContextCache _sharedContextCache =
		new PredictionContextCache();
	public static final int
		If=1, Int=2, IntLiteral=3, StringLiteral=4, AssignmentOP=5, RelationalOP=6, 
		Star=7, Plus=8, Sharp=9, SemiColon=10, Dot=11, Comm=12, LeftBracket=13, 
		RightBracket=14, LeftBrace=15, RightBrace=16, LeftParen=17, RightParen=18, 
		Id=19, Whitespace=20, Newline=21;
	public static String[] channelNames = {
		"DEFAULT_TOKEN_CHANNEL", "HIDDEN"
	};

	public static String[] modeNames = {
		"DEFAULT_MODE"
	};

	private static String[] makeRuleNames() {
		return new String[] {
			"If", "Int", "IntLiteral", "StringLiteral", "AssignmentOP", "RelationalOP", 
			"Star", "Plus", "Sharp", "SemiColon", "Dot", "Comm", "LeftBracket", "RightBracket", 
			"LeftBrace", "RightBrace", "LeftParen", "RightParen", "Id", "Whitespace", 
			"Newline"
		};
	}
	public static final String[] ruleNames = makeRuleNames();

	private static String[] makeLiteralNames() {
		return new String[] {
			null, "'if'", "'int'", null, null, "'='", null, "'*'", "'+'", "'#'", 
			"';'", "'.'", "','", "'['", "']'", "'{'", "'}'", "'('", "')'"
		};
	}
	private static final String[] _LITERAL_NAMES = makeLiteralNames();
	private static String[] makeSymbolicNames() {
		return new String[] {
			null, "If", "Int", "IntLiteral", "StringLiteral", "AssignmentOP", "RelationalOP", 
			"Star", "Plus", "Sharp", "SemiColon", "Dot", "Comm", "LeftBracket", "RightBracket", 
			"LeftBrace", "RightBrace", "LeftParen", "RightParen", "Id", "Whitespace", 
			"Newline"
		};
	}
	private static final String[] _SYMBOLIC_NAMES = makeSymbolicNames();
	public static final Vocabulary VOCABULARY = new VocabularyImpl(_LITERAL_NAMES, _SYMBOLIC_NAMES);

	/**
	 * @deprecated Use {@link #VOCABULARY} instead.
	 */
	@Deprecated
	public static final String[] tokenNames;
	static {
		tokenNames = new String[_SYMBOLIC_NAMES.length];
		for (int i = 0; i < tokenNames.length; i++) {
			tokenNames[i] = VOCABULARY.getLiteralName(i);
			if (tokenNames[i] == null) {
				tokenNames[i] = VOCABULARY.getSymbolicName(i);
			}

			if (tokenNames[i] == null) {
				tokenNames[i] = "<INVALID>";
			}
		}
	}

	@Override
	@Deprecated
	public String[] getTokenNames() {
		return tokenNames;
	}

	@Override

	public Vocabulary getVocabulary() {
		return VOCABULARY;
	}


	public Hello(CharStream input) {
		super(input);
		_interp = new LexerATNSimulator(this,_ATN,_decisionToDFA,_sharedContextCache);
	}

	@Override
	public String getGrammarFileName() { return "Hello.g4"; }

	@Override
	public String[] getRuleNames() { return ruleNames; }

	@Override
	public String getSerializedATN() { return _serializedATN; }

	@Override
	public String[] getChannelNames() { return channelNames; }

	@Override
	public String[] getModeNames() { return modeNames; }

	@Override
	public ATN getATN() { return _ATN; }

	public static final String _serializedATN =
		"\u0004\u0000\u0015y\u0006\uffff\uffff\u0002\u0000\u0007\u0000\u0002\u0001"+
		"\u0007\u0001\u0002\u0002\u0007\u0002\u0002\u0003\u0007\u0003\u0002\u0004"+
		"\u0007\u0004\u0002\u0005\u0007\u0005\u0002\u0006\u0007\u0006\u0002\u0007"+
		"\u0007\u0007\u0002\b\u0007\b\u0002\t\u0007\t\u0002\n\u0007\n\u0002\u000b"+
		"\u0007\u000b\u0002\f\u0007\f\u0002\r\u0007\r\u0002\u000e\u0007\u000e\u0002"+
		"\u000f\u0007\u000f\u0002\u0010\u0007\u0010\u0002\u0011\u0007\u0011\u0002"+
		"\u0012\u0007\u0012\u0002\u0013\u0007\u0013\u0002\u0014\u0007\u0014\u0001"+
		"\u0000\u0001\u0000\u0001\u0000\u0001\u0001\u0001\u0001\u0001\u0001\u0001"+
		"\u0001\u0001\u0002\u0004\u00024\b\u0002\u000b\u0002\f\u00025\u0001\u0003"+
		"\u0001\u0003\u0005\u0003:\b\u0003\n\u0003\f\u0003=\t\u0003\u0001\u0003"+
		"\u0001\u0003\u0001\u0004\u0001\u0004\u0001\u0005\u0001\u0005\u0001\u0005"+
		"\u0001\u0005\u0001\u0005\u0001\u0005\u0003\u0005I\b\u0005\u0001\u0006"+
		"\u0001\u0006\u0001\u0007\u0001\u0007\u0001\b\u0001\b\u0001\t\u0001\t\u0001"+
		"\n\u0001\n\u0001\u000b\u0001\u000b\u0001\f\u0001\f\u0001\r\u0001\r\u0001"+
		"\u000e\u0001\u000e\u0001\u000f\u0001\u000f\u0001\u0010\u0001\u0010\u0001"+
		"\u0011\u0001\u0011\u0001\u0012\u0001\u0012\u0005\u0012e\b\u0012\n\u0012"+
		"\f\u0012h\t\u0012\u0001\u0013\u0004\u0013k\b\u0013\u000b\u0013\f\u0013"+
		"l\u0001\u0013\u0001\u0013\u0001\u0014\u0001\u0014\u0003\u0014s\b\u0014"+
		"\u0001\u0014\u0003\u0014v\b\u0014\u0001\u0014\u0001\u0014\u0001;\u0000"+
		"\u0015\u0001\u0001\u0003\u0002\u0005\u0003\u0007\u0004\t\u0005\u000b\u0006"+
		"\r\u0007\u000f\b\u0011\t\u0013\n\u0015\u000b\u0017\f\u0019\r\u001b\u000e"+
		"\u001d\u000f\u001f\u0010!\u0011#\u0012%\u0013\'\u0014)\u0015\u0001\u0000"+
		"\u0004\u0001\u000009\u0003\u0000AZ__az\u0004\u000009AZ__az\u0002\u0000"+
		"\t\t  \u0081\u0000\u0001\u0001\u0000\u0000\u0000\u0000\u0003\u0001\u0000"+
		"\u0000\u0000\u0000\u0005\u0001\u0000\u0000\u0000\u0000\u0007\u0001\u0000"+
		"\u0000\u0000\u0000\t\u0001\u0000\u0000\u0000\u0000\u000b\u0001\u0000\u0000"+
		"\u0000\u0000\r\u0001\u0000\u0000\u0000\u0000\u000f\u0001\u0000\u0000\u0000"+
		"\u0000\u0011\u0001\u0000\u0000\u0000\u0000\u0013\u0001\u0000\u0000\u0000"+
		"\u0000\u0015\u0001\u0000\u0000\u0000\u0000\u0017\u0001\u0000\u0000\u0000"+
		"\u0000\u0019\u0001\u0000\u0000\u0000\u0000\u001b\u0001\u0000\u0000\u0000"+
		"\u0000\u001d\u0001\u0000\u0000\u0000\u0000\u001f\u0001\u0000\u0000\u0000"+
		"\u0000!\u0001\u0000\u0000\u0000\u0000#\u0001\u0000\u0000\u0000\u0000%"+
		"\u0001\u0000\u0000\u0000\u0000\'\u0001\u0000\u0000\u0000\u0000)\u0001"+
		"\u0000\u0000\u0000\u0001+\u0001\u0000\u0000\u0000\u0003.\u0001\u0000\u0000"+
		"\u0000\u00053\u0001\u0000\u0000\u0000\u00077\u0001\u0000\u0000\u0000\t"+
		"@\u0001\u0000\u0000\u0000\u000bH\u0001\u0000\u0000\u0000\rJ\u0001\u0000"+
		"\u0000\u0000\u000fL\u0001\u0000\u0000\u0000\u0011N\u0001\u0000\u0000\u0000"+
		"\u0013P\u0001\u0000\u0000\u0000\u0015R\u0001\u0000\u0000\u0000\u0017T"+
		"\u0001\u0000\u0000\u0000\u0019V\u0001\u0000\u0000\u0000\u001bX\u0001\u0000"+
		"\u0000\u0000\u001dZ\u0001\u0000\u0000\u0000\u001f\\\u0001\u0000\u0000"+
		"\u0000!^\u0001\u0000\u0000\u0000#`\u0001\u0000\u0000\u0000%b\u0001\u0000"+
		"\u0000\u0000\'j\u0001\u0000\u0000\u0000)u\u0001\u0000\u0000\u0000+,\u0005"+
		"i\u0000\u0000,-\u0005f\u0000\u0000-\u0002\u0001\u0000\u0000\u0000./\u0005"+
		"i\u0000\u0000/0\u0005n\u0000\u000001\u0005t\u0000\u00001\u0004\u0001\u0000"+
		"\u0000\u000024\u0007\u0000\u0000\u000032\u0001\u0000\u0000\u000045\u0001"+
		"\u0000\u0000\u000053\u0001\u0000\u0000\u000056\u0001\u0000\u0000\u0000"+
		"6\u0006\u0001\u0000\u0000\u00007;\u0005\"\u0000\u00008:\t\u0000\u0000"+
		"\u000098\u0001\u0000\u0000\u0000:=\u0001\u0000\u0000\u0000;<\u0001\u0000"+
		"\u0000\u0000;9\u0001\u0000\u0000\u0000<>\u0001\u0000\u0000\u0000=;\u0001"+
		"\u0000\u0000\u0000>?\u0005\"\u0000\u0000?\b\u0001\u0000\u0000\u0000@A"+
		"\u0005=\u0000\u0000A\n\u0001\u0000\u0000\u0000BI\u0005>\u0000\u0000CD"+
		"\u0005>\u0000\u0000DI\u0005=\u0000\u0000EI\u0005<\u0000\u0000FG\u0005"+
		"<\u0000\u0000GI\u0005=\u0000\u0000HB\u0001\u0000\u0000\u0000HC\u0001\u0000"+
		"\u0000\u0000HE\u0001\u0000\u0000\u0000HF\u0001\u0000\u0000\u0000I\f\u0001"+
		"\u0000\u0000\u0000JK\u0005*\u0000\u0000K\u000e\u0001\u0000\u0000\u0000"+
		"LM\u0005+\u0000\u0000M\u0010\u0001\u0000\u0000\u0000NO\u0005#\u0000\u0000"+
		"O\u0012\u0001\u0000\u0000\u0000PQ\u0005;\u0000\u0000Q\u0014\u0001\u0000"+
		"\u0000\u0000RS\u0005.\u0000\u0000S\u0016\u0001\u0000\u0000\u0000TU\u0005"+
		",\u0000\u0000U\u0018\u0001\u0000\u0000\u0000VW\u0005[\u0000\u0000W\u001a"+
		"\u0001\u0000\u0000\u0000XY\u0005]\u0000\u0000Y\u001c\u0001\u0000\u0000"+
		"\u0000Z[\u0005{\u0000\u0000[\u001e\u0001\u0000\u0000\u0000\\]\u0005}\u0000"+
		"\u0000] \u0001\u0000\u0000\u0000^_\u0005(\u0000\u0000_\"\u0001\u0000\u0000"+
		"\u0000`a\u0005)\u0000\u0000a$\u0001\u0000\u0000\u0000bf\u0007\u0001\u0000"+
		"\u0000ce\u0007\u0002\u0000\u0000dc\u0001\u0000\u0000\u0000eh\u0001\u0000"+
		"\u0000\u0000fd\u0001\u0000\u0000\u0000fg\u0001\u0000\u0000\u0000g&\u0001"+
		"\u0000\u0000\u0000hf\u0001\u0000\u0000\u0000ik\u0007\u0003\u0000\u0000"+
		"ji\u0001\u0000\u0000\u0000kl\u0001\u0000\u0000\u0000lj\u0001\u0000\u0000"+
		"\u0000lm\u0001\u0000\u0000\u0000mn\u0001\u0000\u0000\u0000no\u0006\u0013"+
		"\u0000\u0000o(\u0001\u0000\u0000\u0000pr\u0005\r\u0000\u0000qs\u0005\n"+
		"\u0000\u0000rq\u0001\u0000\u0000\u0000rs\u0001\u0000\u0000\u0000sv\u0001"+
		"\u0000\u0000\u0000tv\u0005\n\u0000\u0000up\u0001\u0000\u0000\u0000ut\u0001"+
		"\u0000\u0000\u0000vw\u0001\u0000\u0000\u0000wx\u0006\u0014\u0000\u0000"+
		"x*\u0001\u0000\u0000\u0000\t\u00005;Hdflru\u0001\u0006\u0000\u0000";
	public static final ATN _ATN =
		new ATNDeserializer().deserialize(_serializedATN.toCharArray());
	static {
		_decisionToDFA = new DFA[_ATN.getNumberOfDecisions()];
		for (int i = 0; i < _ATN.getNumberOfDecisions(); i++) {
			_decisionToDFA[i] = new DFA(_ATN.getDecisionState(i), i);
		}
	}
}