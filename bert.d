import std.stdio;

public{
  import std.bigint; // *10-15 times more slow than gmp but this works -- gmp-d conflicts with other deps ((
  import std.conv;
  
/*
// example std.bigint usage
  //BigInt num2 = 1;
  //auto num1 = 1;
  auto num1 = "1";
  BigInt num2 = BigInt(num1);
  num2 += 2;

  string num3 = num2.to!string;
  
  writefln("num2 = %s - %s", num2, typeof(num2).stringof);
  writefln("num3 = %s - %s", num3, typeof(num3).stringof);
  
  BigInt num = 0;
  for(ulong i = 1; i < 4_000_001; i++){
    num += i;
  }
  
  writefln("SUM(1, 4_000_000) = %s", num);

  num = 1;
  for(ubyte j = 1; j < 101; j++){
    num *= j;
  }
  
  writefln("PRODUCT(1, 100) = %s", num);
*/
}

private{
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
}

enum BERT_TAG : ubyte{
  VERSION        = 131,
  SMALL_INT      = 97,
  INT            = 98,
  BIGINT         = 110,
  //LARGE_BIG      = 111,
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
  BigInt,
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
    BigInt bigintValue;
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
      
      case BertType.BigInt:
        return encodeBigInt(bigintValue);
      
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
        
      case BertType.BigInt:
        return bigintValue.to!string;
      
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
          //result.put(format("%02X", b)); // string bytes in hex = <<"blabla">> = <<62,6C,61,62,6C,61>>
          result.put(to!string(b)); // string bytes in dec (like in erlang) = <<"blabla">> = <<98,108,97,98,108,97>>
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

/*
ubyte[] encodeInt(byte value){ // small_integer - int8
  return [cast(ubyte)BERT_TAG.SMALL_INT, cast(ubyte)value];
}
*/

ubyte[] encodeInt(ubyte value){ // small_integer - uint8
  return [cast(ubyte)BERT_TAG.SMALL_INT, value];
}

/*
ubyte[] encodeInt(short value){ // integer - int16
  if(value >= 0 && value <= 255){
    return [cast(ubyte)BERT_TAG.SMALL_INT, cast(ubyte)value];
  }else{
    ubyte[4] bytes = nativeToBigEndian!int(cast(int)value);
    return [cast(ubyte)BERT_TAG.INT] ~ bytes[];
  }
}
*/

ubyte[] encodeInt(ushort value){ // integer - uint16
  if(value <= 255) {
    return [cast(ubyte)BERT_TAG.SMALL_INT, cast(ubyte)value];
  }else{
    ubyte[4] bytes = nativeToBigEndian!int(cast(int)value);
    return [cast(ubyte)BERT_TAG.INT] ~ bytes[];
  }
}

/*
ubyte[] encodeInt(int value){ // integer - int32
  if(value >= 0 && value <= 255){
    return [cast(ubyte)BERT_TAG.SMALL_INT, cast(ubyte)value];
  }else if(value >= -2147483648 && value <= 2147483647){
    ubyte[4] bytes = nativeToBigEndian!int(value);
    return [cast(ubyte)BERT_TAG.INT] ~ bytes[];
  }else{
    return encodeBigInt( cast(ulong)value );
  }
}
*/

ubyte[] encodeInt(uint value){ // integer - uint32
  if(value <= 255){
    return [cast(ubyte)BERT_TAG.SMALL_INT, cast(ubyte)value];
  }else if(value <= 2147483647) {
    ubyte[4] bytes = nativeToBigEndian!int(cast(int)value);
    return [cast(ubyte)BERT_TAG.INT] ~ bytes[];
  }else{
    return encodeBigInt( cast(ulong)value );
  }
}

/*
ubyte[] encodeInt(long value){ // integer - small_big_int - int64 ;; we do not use erlang large_big_int because small_big_int max value = 2^(8*255) - 1 ;; it is too enaught
  if(value >= 0 && value <= 255){
    return [cast(ubyte)BERT_TAG.SMALL_INT, cast(ubyte)value];
  }else if(value >= -2147483648 && value <= 2147483647){
    ubyte[4] bytes = nativeToBigEndian!int(cast(int)value);
    return [cast(ubyte)BERT_TAG.INT] ~ bytes[];
  }else{
    return encodeBigInt( cast(ulong)value );
  }
}
*/

ubyte[] encodeInt(ulong value){ // integer - small_big_int - uint64 ;; we do not use erlang large_big_int because small_big_int max value = 2^(8*255) - 1 ;; it is too enaught
  if(value <= 255){
    return [cast(ubyte)BERT_TAG.SMALL_INT, cast(ubyte)value];
  }else if(value <= 2147483647){
    ubyte[4] bytes = nativeToBigEndian!int(cast(int)value);
    return [cast(ubyte)BERT_TAG.INT] ~ bytes[];
  }else{
    return encodeBigInt(value);
  }
}

/*
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
*/

ubyte[] encodeBigInt(ulong value){
  ubyte[8] temp;
  size_t len = 0;
  
  while(value > 0){
    temp[len++] = cast(ubyte)(value & 0xFF);
    value >>= 8;
  }
  
  if(len == 0){
    return [ cast(ubyte)BERT_TAG.BIGINT, 1, 0, 0 ];
  }
  
  ubyte[] result = [
    cast(ubyte)BERT_TAG.BIGINT,
    cast(ubyte)len,
    0
  ];
  
  foreach_reverse (i; 0..len){
    result ~= temp[i];
  }
  return result;
}

ubyte[] encodeBigInt(BigInt value){
  if(value == 0){
    return [cast(ubyte)BERT_TAG.BIGINT, 1, 0];
  }
  
  //bool isNegative = value < 0; // no sign
  //if(isNegative){
  //  value = -value;
  //}
  
  ubyte[] digits;
  while(value > 0){
    digits ~= cast(ubyte)(value & 0xFF);
    value >>= 8;
  }
  
  /* // we do not use erlang large_big_int because small_big_int max value = 2^(8*255) - 1 ;; it is too enaught
  ubyte tag;
  if(digits.length <= 255) {
    tag = BERT_TAG.BIGINT;
  }else{
    tag = BERT_TAG.LARGE_BIGINT;
  }
  */
  
  ubyte[] result;
  //if(tag == BERT_TAG.BIGINT){
    result = [ cast(ubyte)BERT_TAG.BIGINT,
      cast(ubyte)digits.length,
      //cast(ubyte)(isNegative ? 1 : 0) // no sign
      cast(ubyte)(0) // no sign
    ];
  
  result ~= digits; // in little-endian
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

/*
BertValue bertInt(byte value){
  BertValue v;
  v.type_ = BertType.Int;
  v.intValue = value;
  return v;
}
*/

BertValue bertInt(ubyte value){
  BertValue v;
  v.type_ = BertType.Int;
  v.intValue = value;
  return v;
}

/*
BertValue bertInt(short value){
  BertValue v;
  v.type_ = BertType.Int;
  v.intValue = value;
  return v;
}
*/

BertValue bertInt(ushort value){
  BertValue v;
  v.type_ = BertType.Int;
  v.intValue = value;
  return v;
}

/*
BertValue bertInt(int value){
  BertValue v;
  v.type_ = BertType.Int;
  v.intValue = value;
  return v;
}
*/

BertValue bertInt(uint value){
  BertValue v;
  v.type_ = BertType.Int;
  v.intValue = value;
  return v;
}

/*
BertValue bertInt(long value){
  BertValue v;
  v.type_ = BertType.Int;
  v.intValue = value;
  return v;
}
*/

BertValue bertInt(ulong value){
  BertValue v;
  v.type_ = BertType.Int;
  v.intValue = value;
  return v;
}

BertValue bertBigInt(BigInt value){
  BertValue v;
  v.type_ = BertType.BigInt;
  v.bigintValue = value;
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
    try{
      if(pos >= data.length){ throw new Exception("No data to decode"); }
      if(data[pos++] != BERT_TAG.VERSION){
        throw new Exception("Invalid BERT format: missing version byte");
      }
      
      return decodeValue();
    
    }catch (Exception e){
      writeln("Got error: ", e.msg);
      return bertNil();
    }
  }
  
  private BertValue decodeValue(){
    if(pos >= data.length){ throw new Exception("Unexpected end of data"); }
    ubyte tag = data[pos++];
    
    switch(tag){
      case BERT_TAG.SMALL_INT:
        if(pos >= data.length){ throw new Exception("Incomplete SMALL_INT"); }
        return bertInt( cast(ubyte)data[pos++] );
      
      case BERT_TAG.INT:
        if((pos + 4) > data.length){ throw new Exception("Incomplete INT"); }
        uint value = bigEndianToNative!uint(data[pos..pos+4][0..4]);
        pos += 4;
        
        if(value <= ubyte.max){ // maybe can to ubyte, ushort
          return bertInt( cast(ubyte)value );
        }else if(value <= ushort.max){
          return bertInt( cast(ushort)value );
        }else{ // uint
          return bertInt(value);
        }
      
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
      //case BERT_TAG.LARGE_BIG:
        return decodeBigInt();
      
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
  
  //private BertValue decodeBigInt(ubyte tag){ // BIGINT, not use LARGE_BIG
  private BertValue decodeBigInt(){ // BIGINT, not use LARGE_BIG
    //uint n;
    //ubyte sign; // ignore sign - as unsigned
    
    //if(tag == BERT_TAG.BIGINT){
      if(pos >= data.length){ throw new Exception("Incomplete BIGINT"); }
      uint n = data[pos++];
    //}else{
    //  if((pos + 4) > data.length){ throw new Exception("Incomplete LARGE_BIG size"); }
    //  n = bigEndianToNative!uint(data[pos..pos+4][0..4]);
    //  pos += 4;
    //}
    
    if(pos >= data.length){ throw new Exception("Incomplete BIG sign"); }
    //sign = data[pos++]; // ignore sign - as unsigned
    pos++; // skip sign
    
    if((pos + n) > data.length){ throw new Exception("Incomplete BIG data"); }
    
    if(n <= 8){ // maybe can to ubyte, ushort, uint, ulong
      if(n == 1){
        return bertInt( cast(ubyte)data[pos++] );
      }
      
      ulong value = 0;
      foreach(i; 0..n){
        value += cast(ulong)data[pos++] << (8 * i);
      }
      
      if(n == 2){
        return bertInt( cast(ushort)value );
      }else if( (n == 3) || (n == 4) ){
        return bertInt( cast(uint)value );
      }else{ // ( n == 5) || (n == 6) || (n == 7) || (n == 8)
        return bertInt(value);
      }
      
      //return bertInt(sign ? (-cast(long)value) : cast(long)value);
    
    }else{ // just bigint
      BigInt value = 0;
      
      foreach(i; 0..n){
        value += BigInt(data[pos++]) << (8 * i);
      }
      
      return bertBigInt(value);
    }
  
  }
}

