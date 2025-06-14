# Nom de l'exécutable
TARGET = cube

# Fichiers sources
SRC = main.cpp

# Headers (pour gérer les dépendances)
HEADERS = tiny_obj_loader.h

# Compilateur
CXX = g++

# Options de compilation
CXXFLAGS = -Wall -g

# Bibliothèques à linker
LIBS = -lglfw3 -lglew32 -lopengl32 -lgdi32

# Règle principale
all: $(TARGET)

# Compilation et linkage
$(TARGET): $(SRC) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC) $(LIBS)

# Nettoyage
clean:
	rm -f $(TARGET) *.o

.PHONY: all clean