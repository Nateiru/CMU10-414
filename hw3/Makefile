.PHONY: lib, pybind, clean, format, all

PYTHON3_ROOT_DIR=$(HOME)/miniconda3/envs/cmu
PYTHON3_LIBRARY=$(HOME)/miniconda3/envs/cmu/lib/libpython3.7m.so
PYTHON3_INCLUDE_DIRS=$(HOME)/miniconda3/envs/cmu/include/python3.7m
PYTHON_DEFAULT_EXECUTABLE=$(HOME)/miniconda3/envs/cmu/bin/python

all: lib


lib:
	@echo "PYTHON3_LIBRARY = $(PYTHON3_LIBRARY)"
	@echo "PYTHON3_INCLUDE_DIRS = $(PYTHON3_INCLUDE_DIRS)"
	@echo "PYTHON_DEFAULT_EXECUTABLE = $(PYTHON_DEFAULT_EXECUTABLE)"
	@mkdir -p build
	@cd build; cmake \
			   -DPython_ROOT_DIR=$(PYTHON3_ROOT_DIR) \
			   -DPython_LIBRARY=$(PYTHON3_LIBRARY) \
			   -DPython_INCLUDE_DIRS=$(PYTHON3_INCLUDE_DIRS) \
			   -DPython_DEFAULT_EXECUTABLE=$(PYTHON_DEFAULT_EXECUTABLE) \
			   ..
	@cd build; $(MAKE)

format:
	python3 -m black .
	clang-format -i src/*.cc src/*.cu

clean:
	rm -rf build python/needle/backend_ndarray/ndarray_backend*.so
