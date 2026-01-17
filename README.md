# tenseurpy
Python bindings for tenseur (pythonizations)


## Example usage

- Tensor with gradient information

```python
import tenseur as ten
x = ten.matrix(3, 3, True)
print(x)
print(x.grad())
```
- Basic operations

```python
import tenseur as ten
x = ten.vector(10)
y = ten.vector(10)
z = x + y
print(z.eval())
```


## How to install

Build and install tenseur from source on [https://github.com/istmarc/tenseur](https://github.com/istmarc/tenseur).

```shell
mkdir build-bindings
cd build-bindings
cmake .. -DCMAKE_CXX_COMPILER=clang++ -DTENSEUR_PYTHON=ON
cmake --build . --
sudo make install
```

## Run the tests

Install [pytest](https://docs.pytest.org). Run the tests by using `pytest` on the tenseurpy source directory.

