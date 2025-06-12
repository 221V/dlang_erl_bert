
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

enum BertType{
  Int,
  Float,
  Atom,
  Tuple,
  List,
  Binary,
  Map,
  Nil
}


struct BertValue{
  BertType type_;
  
  union{
    long intValue;
    double floatValue;
    string atomValue;
    BertValue[] tupleValue;
    BertValue[] listValue;
    ubyte[] binaryValue;
    BertValue[BertValue] mapValue;
  }
  
  ubyte[] encode() const{
    final switch(type_){
      case BertType.Int:
        return encodeInt(intValue);
      
      case BertType.Float:
        return encodeFloat(floatValue);
      
      case BertType.Atom:
        return encodeAtom(atomValue);
      
      case BertType.Tuple:
        return encodeTuple(tupleValue.dup);
      
      case BertType.List:
        return encodeList(listValue.dup);
      
      case BertType.Binary:
        return encodeBinary(binaryValue.dup);
      
      case BertType.Map:
        return encodeMap(mapValue.dup);
      
      case BertType.Nil:
        return [cast(ubyte)BERT_TAG.NIL];
    }
  }
  
  string toString() const{
    final switch(type_){
      case BertType.Int:
        return to!string(intValue);
      
      case BertType.Float:
        return to!string(floatValue);
      
      case BertType.Atom:
        return format("'%s'", atomValue);
      
      case BertType.Tuple:
        return "{" ~ tupleValue.map!(e => e.toString()).join(", ") ~ "}";
      
      case BertType.List:
        return "[" ~ listValue.map!(e => e.toString()).join(", ") ~ "]";
      
      case BertType.Binary:
        auto result = appender!string();
        result.put("<<");
        foreach(i, b; binaryValue){
          if(i > 0){ result.put(","); }
          result.put(format("%02X", b));
        }
        result.put(">>");
        return result.data;
      
      case BertType.Map:
        string[] pairs;
        foreach(key, value; mapValue){
          pairs ~= format("%s: %s", key.toString(), value.toString());
        }
        return "#{" ~ pairs.join(", ") ~ "}";
      
      case BertType.Nil:
        return "[]";
    }
  }
}


ubyte[] bertEncode(BertValue term){
  return [cast(ubyte)BERT_TAG.VERSION] ~ term.encode();
}

ubyte[] encodeInt(long value){
  if(value >= 0 && value <= 255){
    return [cast(ubyte)BERT_TAG.SMALL_INT, cast(ubyte)value];
  }else if(value >= -2147483648 && value <= 2147483647){
    ubyte[4] bytes = nativeToBigEndian!int(cast(int)value);
    return [cast(ubyte)BERT_TAG.INT] ~ bytes[];
  }else{
    return encodeBigInt(value);
  }
}

ubyte[] encodeBigInt(long value){
  bool isNegative = (value < 0);
  ulong absValue = isNegative ? (-value) : value;
  
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
  
  foreach_reverse(i; 0..len){
    result ~= temp[i];
  }
  
  return result;
}

ubyte[] encodeFloat(double value){
  ubyte[8] bytes = nativeToBigEndian!double(value);
  return [cast(ubyte)BERT_TAG.FLOAT] ~ bytes[];
}

ubyte[] encodeAtom(string name){
  if(name.length > 255){ throw new Exception("Atom too long"); }
  return [
    cast(ubyte)BERT_TAG.ATOM,
    cast(ubyte)(name.length >> 8),
    cast(ubyte)(name.length & 0xFF)
  ] ~ cast(ubyte[])name;
}

ubyte[] encodeTuple(BertValue[] elements){
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

ubyte[] encodeList(BertValue[] elements){
  ubyte[4] lenBytes = nativeToBigEndian!uint(cast(uint)elements.length);
  return [cast(ubyte)BERT_TAG.LIST] ~ lenBytes[] ~ elements.map!(e => e.encode()).join() ~ cast(ubyte)BERT_TAG.NIL;
}

ubyte[] encodeBinary(ubyte[] data){
  ubyte[4] lenBytes = nativeToBigEndian!uint(cast(uint)data.length);
  return [cast(ubyte)BERT_TAG.BINARY] ~ lenBytes[] ~ data;
}

ubyte[] encodeMap(const BertValue[BertValue] map){
  ubyte[4] lenBytes = nativeToBigEndian!uint(cast(uint)map.length);
  return [cast(ubyte)BERT_TAG.MAP] ~ lenBytes[] ~ map.byPair.map!(p => p.key.encode() ~ p.value.encode()).join();
}

BertValue bertInt(long value){
  BertValue v;
  v.type_ = BertType.Int;
  v.intValue = value;
  return v;
}

BertValue bertFloat(double value){
  BertValue v;
  v.type_ = BertType.Float;
  v.floatValue = value;
  return v;
}

BertValue bertAtom(string name){
  BertValue v;
  v.type_ = BertType.Atom;
  v.atomValue = name;
  return v;
}

BertValue bertBinary(ubyte[] data){
  BertValue v;
  v.type_ = BertType.Binary;
  v.binaryValue = data;
  return v;
}

BertValue bertList(BertValue[] elements){
  BertValue v;
  v.type_ = BertType.List;
  v.listValue = elements;
  return v;
}

BertValue bertTuple(BertValue[] elements){
  BertValue v;
  v.type_ = BertType.Tuple;
  v.tupleValue = elements;
  return v;
}

BertValue bertMap(BertValue[BertValue] map){
  BertValue v;
  v.type_ = BertType.Map;
  v.mapValue = map;
  return v;
}

BertValue bertNil(){
  BertValue v;
  v.type_ = BertType.Nil;
  return v;
}


struct BertDecoder{
  ubyte[] data;
  size_t pos;
  
  BertValue decode(){
    if(pos >= data.length){ throw new Exception("No data to decode"); }
    if(data[pos++] != BERT_TAG.VERSION){
      throw new Exception("Invalid BERT format: missing version byte");
    }
    return decodeValue();
  }
  
  private BertValue decodeValue(){
    if(pos >= data.length){ throw new Exception("Unexpected end of data"); }
    
    ubyte tag = data[pos++];
    
    switch(tag){
      case BERT_TAG.SMALL_INT:
        if(pos >= data.length){ throw new Exception("Incomplete SMALL_INT"); }
        return bertInt(data[pos++]);
      
      case BERT_TAG.INT:
        if((pos + 4) > data.length){ throw new Exception("Incomplete INT"); }
        int value = bigEndianToNative!int(data[pos..pos+4][0..4]);
        pos += 4;
        return bertInt(value);
      
      case BERT_TAG.FLOAT:
        if((pos + 8) > data.length){ throw new Exception("Incomplete FLOAT"); }
        double fvalue = bigEndianToNative!double(data[pos..pos+8][0..8]);
        pos += 8;
        return bertFloat(fvalue);
      
      case BERT_TAG.ATOM:
        if((pos + 2) > data.length){ throw new Exception("Incomplete ATOM length"); }
        ushort len = (cast(ushort)data[pos] << 8) | data[pos+1];
        pos += 2;
        if(pos + len > data.length){ throw new Exception("Incomplete ATOM data"); }
        string atom = cast(string)data[pos..pos+len];
        pos += len;
        return bertAtom(atom);
      
      case BERT_TAG.TUPLE:
        if(pos >= data.length){ throw new Exception("Incomplete TUPLE"); }
        ubyte arity = data[pos++];
        auto elements = new BertValue[arity];
        foreach(i; 0..arity){
          elements[i] = decodeValue();
        }
        return bertTuple(elements);
      
      case BERT_TAG.LIST:
        if((pos + 4) > data.length){ throw new Exception("Incomplete LIST length"); }
        uint len = bigEndianToNative!uint(data[pos..pos+4][0..4]);
        pos += 4;
        auto elements = new BertValue[len];
        foreach(i; 0..len){
          elements[i] = decodeValue();
        }
        
        if(pos >= data.length || data[pos++] != BERT_TAG.NIL){
          throw new Exception("Missing NIL terminator for LIST");
        }
        return bertList(elements);
      
      case BERT_TAG.BINARY:
        if((pos + 4) > data.length){ throw new Exception("Incomplete BINARY length"); }
        uint len = bigEndianToNative!uint(data[pos..pos+4][0..4]);
        pos += 4;
        if((pos + len) > data.length){ throw new Exception("Incomplete BINARY data"); }
        auto bin = bertBinary(data[pos..pos+len]);
        pos += len;
        return bin;
      
      case BERT_TAG.NIL:
        return bertNil();
      
      case BERT_TAG.BIGINT:
      case BERT_TAG.LARGE_BIG:
        return decodeBigInt(tag);
      
      case BERT_TAG.MAP:
        if((pos + 4) > data.length){ throw new Exception("Incomplete MAP size"); }
        uint size = bigEndianToNative!uint(data[pos..pos+4][0..4]);
        pos += 4;
        BertValue[BertValue] map;
        foreach(i; 0..size){
          auto key = decodeValue();
          auto value = decodeValue();
          map[key] = value;
        }
        return bertMap(map);
      
      default:
        throw new Exception(format("Unknown BERT tag: 0x%x", tag));
    }
  }
  
  private BertValue decodeBigInt(ubyte tag){
    uint n;
    ubyte sign;
    
    if(tag == BERT_TAG.BIGINT){
      if(pos >= data.length){ throw new Exception("Incomplete BIGINT"); }
      n = data[pos++];
    }else{
      if((pos + 4) > data.length){ throw new Exception("Incomplete LARGE_BIG size"); }
      n = bigEndianToNative!uint(data[pos..pos+4][0..4]);
      pos += 4;
    }
    
    if(pos >= data.length){ throw new Exception("Incomplete BIG sign"); }
    sign = data[pos++];
    
    if((pos + n) > data.length){ throw new Exception("Incomplete BIG data"); }
    ulong value = 0;
    
    foreach_reverse(i; 0..n){
      value = (value << 8) | data[pos++];
    }
    
    return bertInt(sign ? (-cast(long)value) : cast(long)value);
  }
}

