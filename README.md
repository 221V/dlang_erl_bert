# D dlang Erlang BERT encode && decode data

```
%% erlang

erlang:term_to_binary(X).
erlang:binary_to_term(Y).

> X = { test, 42, 3.14159, [1, 2, 3], <<222, 173, 190, 239>> }.
> io:format("~p~n", [term_to_binary(X)]).
<<131,104,5,100,0,4,116,101,115,116,97,42,70,64,9,33,249,240,27,134,110,107,0,3,1,2,3,109,0,0,0,4,222,173,190,239>>
```

```
// js (check bert.js -- code from n2o v4.4 with fixes)

enc(tuple( atom('test'),  number(42), float(3.14159), list(1, 2, 3), bin( new Uint8Array([222, 173, 190, 239]) ) ) ); // bin('blabla'), bignum(bigInt(number))
// Uint8Array(39)[131,104,5,118,0,4,116,101,115,116,97,42,70,64,9,33,249,240,27,134,110,108,0,0,0,3,106,106,106,106,109,0,0,0,4,222,173,190,239]
```

```
// dlang oop

import ber_oop;

void main(){
  // encode
  auto data = bertTuple([
    bertAtom("test"),
    bertInt(42),
    bertFloat(3.14159),
    bertList([
      bertInt(1),
      bertInt(2),
      bertInt(3)
    ]),
    bertBinary([0xDE, 0xAD, 0xBE, 0xEF])
  ]);
  
  ubyte[] encoded = bertEncode(data);
  writeln("Encoded: ", encoded);
  
  // decode
  auto decoder = new BertDecoder(encoded);
  auto decoded = decoder.decode();
  writeln("Decoded: ", decoded.toString());
}
/*
Encoded: [131, 104, 5, 118, 0, 4, 116, 101, 115, 116, 97, 42, 70, 64, 9, 33, 249, 240, 27, 134, 110, 108, 0, 0, 0, 3, 97, 1, 97, 2, 97, 3, 106, 109, 0, 0, 0, 4, 222, 173, 190, 239]
Decoded: {'test', 42, 3.14159, [1, 2, 3], <<DE,AD,BE,EF>>}
*/
```


```
// dlang fp

import bert;

void main(){
  // encode
  auto data = bertTuple([
    bertAtom("test"),
    bertInt(42),
    bertFloat(3.14159),
    bertList([
      bertInt(1),
      bertInt(2),
      bertInt(3)
    ]),
    bertBinary([0xDE, 0xAD, 0xBE, 0xEF])
  ]);
  
  ubyte[] encoded = bertEncode(data);
  writeln("Encoded: ", encoded);
  
  // decode
  auto decoder = BertDecoder(encoded);
  auto decoded = decoder.decode();
  writeln("Decoded: ", decoded.toString());
}

/*
Encoded: [131, 104, 5, 118, 0, 4, 116, 101, 115, 116, 97, 42, 70, 64, 9, 33, 249, 240, 27, 134, 110, 108, 0, 0, 0, 3, 97, 1, 97, 2, 97, 3, 106, 109, 0, 0, 0, 4, 222, 173, 190, 239]
Decoded: {'test', 42, 3.14159, [1, 2, 3], <<DE,AD,BE,EF>>}
*/
```


```
// https://www.erlang.org/docs/25/apps/erts/erl_ext_dist.html
// https://www.erlang.org/doc/apps/erts/erl_ext_dist.html
  SMALL_INTEGER_EXT = 97 // 1 = Int // Unsigned 8-bit integer
  INTEGER_EXT = 98 // 4 = Int // Signed 32-bit integer in big-endian format
  FLOAT_EXT = 99 // 31 = Float string
  NEW_FLOAT_EXT = 70 // 8 = IEEE float // 8 bytes in big-endian IEEE format
  
  SMALL_TUPLE_EXT = 104 // 1 = Arity, N = Elements // Arity field is an unsigned byte that determines how many elements that follows in section Elements
  LARGE_TUPLE_EXT = 105 // 4 = Arity, N = Elements // Arity is an unsigned 4 byte integer in big-endian format
  MAP_EXT = 116 // 4 = Arity, N = Pairs // Arity field is an unsigned 4 byte integer in big-endian format that determines the number of key-value pairs in the map
    // Key and value pairs (Ki => Vi) are encoded in section Pairs in the following order: K1, V1, K2, V2,..., Kn, Vn
    // Duplicate keys are not allowed within the same map
  
  NIL_EXT = 106 // empty list, that is, the Erlang syntax []
  STRING_EXT = 107 // 2 = Len, Len = Characters // lists of bytes (integer in the range 0-255) // field Len is an unsigned 2 byte integer (big-endian)
    //implementations must ensure that lists longer than 65535 elements are encoded as LIST_EXT
  LIST_EXT = 108 // 4 = Len, Elements, Tail // Len is the number of elements that follows in section Elements
    // Tail is the final tail of the list; it is NIL_EXT for a proper list, but can be any type if the list is improper (for example, [a|b])
  
  BINARY_EXT = 109 // 4 = Len, Len = Data // Len length field is an unsigned 4 byte integer (big-endian)
    // Binaries are generated with bit syntax expression or with erlang:list_to_binary/1, erlang:term_to_binary/1, or as input from binary ports
  
  SMALL_BIG_EXT = 110 // 1 = n, 1 = Sign, n = d(0) ... d(n-1)
    // Bignums are stored in unary form with a Sign byte, that is, 0 if the bignum is positive and 1 if it is negative.
    // The digits are stored with the least significant byte stored first.
    // To calculate the integer, the following formula can be used:
    // B = 256
    // (d0*B^0 + d1*B^1 + d2*B^2 + ... d(N-1)*B^(n-1))
  LARGE_BIG_EXT = 111 // 4 = n, 1 = Sign, n = d(0) ... d(n-1)
  
  ATOM_UTF8_EXT = 118 // 2 = Len, Len = AtomName
  SMALL_ATOM_UTF8_EXT = 119 // 1 = Len, Len = AtomName
  ATOM_EXT (deprecated) = 100 // 2 = Len, Len = AtomName // 2 byte unsigned length in big-endian order, followed by Len numbers of 8-bit Latin-1 characters
    // that forms the AtomName
    // The maximum allowed value for Len is 255
  SMALL_ATOM_EXT (deprecated) = 115 // 1 = Len, Len = AtomName // 1 byte unsigned length, followed by Len numbers of 8-bit Latin-1 characters that forms the AtomName
  
  BIT_BINARY_EXT = 77 // 4 = Len, 1 = Bits, Len = Data
    // bitstring whose length in bits does not have to be a multiple of 8
    // Len field is an unsigned 4 byte integer (big-endian)
    // Bits field is the number of bits (1-8) that are used in the last byte in the data field, counting from the most significant bit to the least significant
  
  ATOM_CACHE_REF = 82
  PORT_EXT = 102
  NEW_PORT_EXT = 89
  V4_PORT_EXT = 120
  PID_EXT = 103
  NEW_PID_EXT = 88
  REFERENCE_EXT (deprecated) = 101
  NEW_REFERENCE_EXT = 114
  NEWER_REFERENCE_EXT = 90
  FUN_EXT (removed) = 117
  NEW_FUN_EXT = 112
  EXPORT_EXT = 113
  LOCAL_EXT = 121
```

#### todo more examples, todo tests

