
using System.Numerics;

namespace blazingzephyr.itmo.tensors;

public ref struct Tensor<T> where T : unmanaged, INumberBase<T>
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
}
