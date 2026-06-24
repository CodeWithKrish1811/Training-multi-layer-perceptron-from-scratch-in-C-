# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -O3

# Math library linking is required on Linux/macOS
LDFLAGS = -lm

# Executable name
TARGET = nn_xor

# Detect OS to handle clean commands and executable extensions
ifeq ($(OS),Windows_NT)
    EXEC = $(TARGET).exe
    RM = del /Q
    FIX_PATH = $(subst /,\,$1)
else
    EXEC = $(TARGET)
    RM = rm -f
    FIX_PATH = $1
endif

# Source and Object files
SRCS = main.c matrix.c nn.c
OBJS = $(SRCS:.c=.o)

.PHONY: all clean

# Default rule
all: $(EXEC)

# Link the executable
$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(EXEC) $(LDFLAGS)

# Compile source files into object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean compile artifacts
clean:
	$(RM) $(call FIX_PATH,$(OBJS)) $(call FIX_PATH,$(EXEC))
