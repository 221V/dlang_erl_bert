
import std.stdio;
import std.conv;
import std.bitmanip;
import std.string;
import std.utf;
import std.math;
import std.traits;
import std.algorithm;
import std.range;
import std.digest : toHexString;
import std.array : appender;
import std.format : format;

enum BERT_TAG : ubyte{
  VERSION        = 131,
  SMALL_INT      = 97,
  INT            = 98,
  BIGINT         = 110,
  LARGE_BIG      = 111,
  FLOAT          = 70,
  ATOM           = 118,
  TUPLE          = 104, // SMALL_TUPLE
  LARGE_TUPLE    = 105,
  NIL            = 106,
  LIST           = 108,
  BINARY         = 109, // use BINARY as STRING
  MAP            = 116
}

abstract class BertTerm{
  abstract ubyte[] encode();
  abstract override string toString();
}

ubyte[] bertEncode(BertTerm term){
  return [cast(ubyte)BERT_TAG.VERSION] ~ term.encode();
}

class BertInt : BertTerm{
  long value;
  
  this(long v){ value = v; }
  
  override ubyte[] encode(){
    if(value >= 0 && value <= 255){
      return [cast(ubyte)BERT_TAG.SMALL_INT, cast(ubyte)value];
    }else if(value >= -2147483648 && value <= 2147483647){
      ubyte[4] bytes = nativeToBigEndian!int(cast(int)value);
      return [cast(ubyte)BERT_TAG.INT] ~ bytes[];
    }else{
      return encodeBigInt();
    }
  }
  
  private ubyte[] encodeBigInt(){
    bool isNegative = value < 0;
    ulong absValue = isNegative ? -value : value;
    
    ubyte[8] temp;
    size_t len = 0;
    while(absValue > 0){
      temp[len++] = cast(ubyte)(absValue & 0xFF);
      absValue >>= 8;
    }
    
    ubyte[] result = [
      cast(ubyte)BERT_TAG.BIGINT,
      cast(ubyte)len,
      isNegative ? 1 : 0
    ];
    
    foreach_reverse (i; 0..len){
      result ~= temp[i];
    }
    
    return result;
  }
  
  override string toString(){
    return to!string(value);
  }
}

class BertFloat : BertTerm{
  double value;
  
  this(double v){ value = v; }
  
  override ubyte[] encode(){
    ubyte[8] bytes = nativeToBigEndian!double(value);
    return [cast(ubyte)BERT_TAG.FLOAT] ~ bytes[];
  }
  
  override string toString(){
    return to!string(value);
  }
}

class BertAtom : BertTerm{ // atom
  string name;
  
  this(string n){
    if(n.length > 255){ throw new Exception("Atom too long"); }
    name = n;
  }
  
  override ubyte[] encode(){
    return [
      cast(ubyte)BERT_TAG.ATOM,
      cast(ubyte)(name.length >> 8),
      cast(ubyte)(name.length & 0xFF)
    ] ~ cast(ubyte[])name;
  }
  
  override string toString(){
    return format("'%s'", name);
  }
}

class BertTuple : BertTerm{ // tuple
  BertTerm[] elements;
  
  this(BertTerm[] el){
    elements = el;
  }
  
  override ubyte[] encode(){
    ubyte[] result;
    if(elements.length <= 255){
      result = [cast(ubyte)BERT_TAG.TUPLE, cast(ubyte)elements.length];
    }else{
      result = [cast(ubyte)BERT_TAG.LARGE_TUPLE] ~ nativeToBigEndian!uint(cast(uint)elements.length);
    }
    
    foreach(elem; elements){
      result ~= elem.encode();
    }
    
    return result;
  }
  
  override string toString(){
    return "{" ~ elements.map!(e => e.toString()).join(", ") ~ "}";
  }
}

class BertList : BertTerm{ // list with elements same type
  BertTerm[] elements;
  
  this(BertTerm[] el){
    elements = el;
  }
  
  override ubyte[] encode(){
    ubyte[4] lenBytes = nativeToBigEndian!uint(cast(uint)elements.length);
    ubyte[] result = [
      cast(ubyte)BERT_TAG.LIST
    ] ~ lenBytes[] ~ elements.map!(e => e.encode()).join() ~ cast(ubyte)BERT_TAG.NIL;
    return result;
  }
  
  override string toString(){
    return "[" ~ elements.map!(e => e.toString()).join(", ") ~ "]";
  }
}

class BertBinary : BertTerm{ // binary string
  ubyte[] data;
  
  this(ubyte[] d){
    data = d;
  }
  
  override ubyte[] encode(){
    ubyte[4] lenBytes = nativeToBigEndian!uint(cast(uint)data.length);
    return [cast(ubyte)BERT_TAG.BINARY] ~ lenBytes[] ~ data;
  }
  
  override string toString(){
    auto result = appender!string();
    result.put("<<");
    foreach(i, b; data){
      if(i > 0){ result.put(","); }
      //result.put(format("%02X", b)); // string bytes in hex = <<"blabla">> = <<62,6C,61,62,6C,61>>
      result.put(to!string(b)); // string bytes in dec (like in erlang) = <<"blabla">> = <<98,108,97,98,108,97>>
    }
    result.put(">>");
    return result.data; // "<<binary data>>"
  }
}

class BertMap : BertTerm{ // map
  BertTerm[BertTerm] map;
  
  this(BertTerm[BertTerm] m){
    map = m;
  }
  
  override ubyte[] encode(){
    ubyte[4] lenBytes = nativeToBigEndian!uint(cast(uint)map.length);
    ubyte[] result = [
      cast(ubyte)BERT_TAG.MAP
    ] ~ lenBytes[] ~ map.byPair.map!(p => p.key.encode() ~ p.value.encode()).join();
    return result;
  }
  
  override string toString(){
    string[] pairs;
    foreach(key, value; map){
      pairs ~= format("%s: %s", key.toString(), value.toString());
    }
    return "#{" ~ pairs.join(", ") ~ "}";
  }
}


class BertDecoder{ // BERT Decoder
  ubyte[] data;
  size_t pos;
  
  this(ubyte[] d){
    data = d;
    pos = 0;
  }
  
  BertTerm decode(){
    if(pos >= data.length) throw new Exception("No data to decode");
    if(data[pos++] != BERT_TAG.VERSION){
      throw new Exception("Invalid BERT format: missing version byte");
    }
    return decodeValue();
  }
  
  private BertTerm decodeValue(){
    if(pos >= data.length) throw new Exception("Unexpected end of data");
    
    ubyte tag = data[pos++];
    
    switch(tag){
      case BERT_TAG.SMALL_INT:
        if(pos >= data.length) throw new Exception("Incomplete SMALL_INT");
        return new BertInt(data[pos++]);
      
      case BERT_TAG.INT:
        if(pos + 4 > data.length) throw new Exception("Incomplete INT");
        int value = bigEndianToNative!int(data[pos..pos+4][0..4]);
        pos += 4;
        return new BertInt(value);
      
      case BERT_TAG.FLOAT:
        if(pos + 8 > data.length) throw new Exception("Incomplete FLOAT");
        double fvalue = bigEndianToNative!double(data[pos..pos+8][0..8]);
        pos += 8;
        return new BertFloat(fvalue);
      
      case BERT_TAG.ATOM:
        if(pos + 2 > data.length) throw new Exception("Incomplete ATOM length");
        ushort len = (cast(ushort)data[pos] << 8) | data[pos+1];
        pos += 2;
        if(pos + len > data.length) throw new Exception("Incomplete ATOM data");
        string atom = cast(string)data[pos..pos+len];
        pos += len;
        return new BertAtom(atom);
      
      case BERT_TAG.TUPLE:
        if(pos >= data.length) throw new Exception("Incomplete TUPLE");
        ubyte arity = data[pos++];
        auto elements = new BertTerm[arity];
        foreach(i; 0..arity){
          elements[i] = decodeValue();
        }
        return new BertTuple(elements);
      
      case BERT_TAG.LIST:
        if(pos + 4 > data.length) throw new Exception("Incomplete LIST length");
        uint len = bigEndianToNative!uint(data[pos..pos+4][0..4]);
        pos += 4;
        auto elements = new BertTerm[len];
        foreach(i; 0..len){
          elements[i] = decodeValue();
        }
        
        if(pos >= data.length || data[pos++] != BERT_TAG.NIL){
          throw new Exception("Missing NIL terminator for LIST");
        }
        return new BertList(elements);
      
      case BERT_TAG.BINARY:
        if(pos + 4 > data.length) throw new Exception("Incomplete BINARY length");
        uint len = bigEndianToNative!uint(data[pos..pos+4][0..4]);
        pos += 4;
        if(pos + len > data.length) throw new Exception("Incomplete BINARY data");
        auto bin = new BertBinary(data[pos..pos+len]);
        pos += len;
        return bin;
      
      case BERT_TAG.NIL:
        return new BertList([]);
      
      case BERT_TAG.BIGINT:
      case BERT_TAG.LARGE_BIG:
        return decodeBigInt(tag);
      
      case BERT_TAG.MAP:
        if(pos + 4 > data.length) throw new Exception("Incomplete MAP size");
        uint size = bigEndianToNative!uint(data[pos..pos+4][0..4]);
        pos += 4;
        auto map = new BertTerm[BertTerm];
        foreach(i; 0..size){
          auto key = decodeValue();
          auto value = decodeValue();
          map[key] = value;
        }
        return new BertMap(map);
      
      default:
        throw new Exception(format("Unknown BERT tag: 0x%x", tag));
    }
  }
  
  private BertTerm decodeBigInt(ubyte tag){
    uint n;
    ubyte sign;
    
    if(tag == BERT_TAG.BIGINT){
      if(pos >= data.length) throw new Exception("Incomplete BIGINT");
      n = data[pos++];
    }else{
      if(pos + 4 > data.length) throw new Exception("Incomplete LARGE_BIG size");
      n = bigEndianToNative!uint(data[pos..pos+4][0..4]);
      pos += 4;
    }
    
    if(pos >= data.length) throw new Exception("Incomplete BIG sign");
    sign = data[pos++];
    
    if(pos + n > data.length) throw new Exception("Incomplete BIG data");
    ulong value = 0;
    
    foreach_reverse(i; 0..n){
      value = (value << 8) | data[pos++];
    }
    
    return new BertInt(sign ? -cast(long)value : cast(long)value);
  }
}


BertTerm bertInt(long value){ return new BertInt(value); }
BertTerm bertFloat(double value){ return new BertFloat(value); }
BertTerm bertAtom(string name){ return new BertAtom(name); }
BertTerm bertBinary(ubyte[] data){ return new BertBinary(data); }
BertTerm bertList(BertTerm[] elements){ return new BertList(elements); }
BertTerm bertTuple(BertTerm[] elements){ return new BertTuple(elements); }
BertTerm bertMap(BertTerm[BertTerm] map){ return new BertMap(map); }

