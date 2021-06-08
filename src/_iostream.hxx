#pragma once
#include <string>
#include <vector>
#include <ostream>
#include <fstream>
#include <iostream>

using std::ios;
using std::string;
using std::vector;
using std::ostream;
using std::ifstream;
using std::cout;




// READ-FILE
// ---------

string readFile(const char *pth) {
  string a; ifstream f(pth);
  f.seekg(0, ios::end);
  a.resize(f.tellg());
  f.seekg(0);
  f.read((char*) a.data(), a.size());
  return a;
}




// WRITE
// -----

template <class T>
void write(ostream& a, const T *x, int N) {
  a << "{";
  for (int i=0; i<N; i++)
    a << " " << x[i];
  a << " }";
}

template <class T>
void write(ostream& a, const vector<T>& x) {
  write(a, x.data(), x.size());
}

template <class T>
void write(ostream& a, const vector<vector<T>>& x) {
  a << "{\n";
  for (auto& v : x) {
    a << "  "; write(a, v);
  }
  a << "}";
}


template <class G>
void write(ostream& a, const G& x, bool all=false) {
  a << "order: " << x.order() << " size: " << x.size();
  if (!all) { a << " {}"; return; }
  a << " {\n";
  for (int u : x.vertices()) {
    a << "  " << u << " ->";
    for (int v : x.edges(u))
      a << " " << v;
    a << "\n";
  }
  a << "}";
}




// PRINT
// -----

template <class T>
void print(const T *x, int N) { write(cout, x, N); }

template <class T>
void print(const vector<T>& x) { write(cout, x); }

template <class T>
void print(const vector<vector<T>>& x) { write(cout, x); }


template <class G>
void print(const G& x, bool all=false) { write(cout, x, all); }




// PRINTLN
// -------

template <class T>
void println(const T *x, int N) { print(x, N); cout << "\n"; }

template <class T>
void println(const vector<T>& x) { print(x); cout << "\n"; }

template <class T>
void println(const vector<vector<T>>& x) { print(x); cout << "\n"; }


template <class G>
void println(const G& x, bool all=false) { print(x, all); cout << "\n"; }
