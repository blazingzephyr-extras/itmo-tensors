
namespace blazingzephyr.itmo.tensors.impl1;

public ref struct Tensor<T> where T : unmanaged, System.Numerics.INumberBase<T>
{
    private readonly int[] _shape;
    private readonly int[] _strides;
    private Span<T> _data;

    public Tensor(Span<T> data, params int[] shape)
    {
        _data = data;
        _shape = shape;
        _strides = CalculateStrides(shape);
    }

    public ref T this[ReadOnlySpan<int> indices] => ref _data[FlatIndex(indices)];
    public ref T this[params int[] indices] => ref _data[FlatIndex(indices)];

    private Tensor(T[] data, int[] shape, int[] strides)
    {
        _data = data;
        _shape = shape;
        _strides = strides;
    }

    private static int[] CalculateStrides(int[] shape)
    {
        int[] strides = new int[shape.Length];
        strides[^1] = 1;
        for (int i = shape.Length - 2; i >= 0; i--)
            strides[i] = strides[i + 1] * shape[i + 1];

        return strides;
    }

    public readonly int FlatIndex(ReadOnlySpan<int> indices)
    {
        int result = 0;
        for (int i = 0; i < indices.Length; i++)
        {
            result += indices[i] * _strides[i];
        }

        return result;
    }

    public static Tensor<T> Add(Tensor<T> a, T ac, Tensor<T> b, T bc, Span<T> dest)
    {
        if (!a._shape.SequenceEqual(b._shape))
        {
            throw new ArgumentException(
                "Сложить можно только два тензора с одинаковой формой/размерностью");
        }

        Tensor<T> result = new Tensor<T>(dest, a._shape);
        for (int i = 0; i < dest.Length; i++)
        {
            dest[i] = a._data[i] * ac + b._data[i] * bc;
        }

        return result;
    }

    /// <summary>
    /// Тензорное произведение тензоров.
    /// </summary>
    /// <param name="a"></param>
    /// <param name="b"></param>
    /// <param name="dest"></param>
    public static Tensor<T> OuterProduct(Tensor<T> a, Tensor<T> b, Span<T> dest)
    {
        int[] shape = [.. a._shape, .. b._shape];
        Tensor<T> result = new Tensor<T>(dest, shape);
        Span<int> ai = stackalloc int[a._shape.Length];
        Span<int> bi = stackalloc int[b._shape.Length];
        for (int flat = 0; flat < dest.Length; flat++)
        {
            int i = flat;
            for (int j = 0; j < a._shape.Length; j++)
            {
                ai[j] = i / result._strides[j];
                i %= result._strides[j];
            }
            for (int j = 0; j < b._shape.Length; j++)
            {
                bi[j] = i / result._strides[j + a._shape.Length];
                i %= result._strides[j + a._shape.Length];
            }
            dest[flat] = a._data[a.FlatIndex(ai)] * b._data[b.FlatIndex(bi)];
        }
        return result;
    }

    /// <summary>
    /// Произведение Кронекера - операция между двумя матрицами,
    /// подвид тензорного произведения,
    /// результатом которого является блочная матрица.
    /// </summary>
    /// <param name="a">Левый элемент произведения.</param>
    /// <param name="b">Правый элемент произведения.</param>
    /// <param name="dest">Представление данных,
    /// куда требуется записать новые компоненты</param>
    /// <returns>Тензор, представляющий блочную матрицу.</returns>
    public static Tensor<T> KroneckerProduct(Tensor<T> a, Tensor<T> b, Span<T> dest)
    {
        if (a._shape.Length != b._shape.Length)
        {
            throw new InvalidOperationException(
                "Произведение Кронекера можно применить только к матрицам." +
                "Для обобщения в этой программе также можно применить произведение Кронекера" +
                "к тензорам одинаковой размерности");
        }

        int rank = a._shape.Length;
        int[] shape = new int[rank];
        for (int i = 0; i < shape.Length; i++)
        {
            shape[i] = a._shape[i] * b._shape[i];
        }

        Tensor<T> result = new Tensor<T>(dest, shape);
        Span<int> ai = stackalloc int[rank];
        Span<int> bi = stackalloc int[rank];

        for (int flat = 0; flat < dest.Length; flat++)
        {
            int rem = flat;
            for (int d = rank - 1; d >= 0; d--)
            {
                int idx = rem % shape[d];
                ai[d] = idx / b._shape[d];  // номер блока A
                bi[d] = idx % b._shape[d];  // позиция внутри блока B
                rem /= shape[d];
            }

            dest[flat] = a._data[a.FlatIndex(ai)] * b._data[b.FlatIndex(bi)];
        }

        return result;
    }
}
