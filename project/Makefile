# Nom de l'exécutable
TARGET = cube

# Fichiers sources
SRC = main.cpp

# Compilateur
CXX = g++

# Options de compilation
CXXFLAGS = -Wall -g

# Bibliothèques à linker
LIBS = -lglfw3 -lglew32 -lopengl32 -lgdi32

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean
