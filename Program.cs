
namespace blazingzephyr.itmo.tensors;

internal class Program
{
    static void Main()
    {
        Console.WriteLine("Примитивные тензорные / SIMD операции из .NET8:");
        Console.ReadKey();
        Example1();

        Console.WriteLine("----------");
        Console.ReadKey();
        Example2();

        Console.WriteLine();
        Console.WriteLine("Собственная имплементация тензоров:");
        Console.ReadKey();
        OuterProduct1();

        Console.WriteLine();
        Console.WriteLine("Собственная имплементация тензоров:");
        Console.ReadKey();
        OuterProduct2();
    }

    static void Example1()
    {
        // Встроенная имплементация из .NET 8 (ноябрь 2023)
        // Примитивные операции с тензорами.
        //
        ReadOnlySpan<double> x = [10.0, 5.0, 6.0, 8.0, 9.0];
        Console.WriteLine(String.Join(", ", x.ToArray()));

        double y = 8.0;
        Console.WriteLine($" + {y}");

        Span<double> result = stackalloc double[5];
        System.Numerics.Tensors.TensorPrimitives.Add(x, y, result);

        Console.WriteLine();
        Console.WriteLine(String.Join(", ", result.ToArray()));
    }

    static void Example2()
    {
        // Встроенная имплементация из .NET 8 (ноябрь 2023)
        // Примитивные операции с тензорами.
        //
        ReadOnlySpan<System.Numerics.Complex> x = [
            new System.Numerics.Complex(11.0, 4.0),
            new System.Numerics.Complex(2.0, 4.0),
            new System.Numerics.Complex(5.0, 0.0)
        ];
        Console.WriteLine(String.Join(", ", x.ToArray()));
        Console.WriteLine("Проверка Span<T> на комплексность.");

        Span<bool> result = stackalloc bool[3];
        System.Numerics.Tensors.TensorPrimitives.IsComplexNumber(x, result);

        Console.WriteLine();
        Console.WriteLine(String.Join(", ", result.ToArray()));
    }

    static void OuterProduct1()
    {
        Span<double> a = [1, 0, 0, 1];
        impl1.Tensor<double> A = new impl1.Tensor<double>(a, 2, 2);

        Span<double> b = [1, 1, 1, -1];
        impl1.Tensor<double> B = new impl1.Tensor<double>(b, 2, 2);

        Span<double> c = stackalloc double[2 * 2 * 2 * 2];
        impl1.Tensor<double> C = impl1.Tensor<double>.OuterProduct(A, B, c);

        Console.WriteLine(C[0, 0, 1, 1]);
    }

    static void OuterProduct2()
    {
        Span<double> a = [1, 2, 3, 4];
        impl1.Tensor<double> A = new impl1.Tensor<double>(a, 4, 1);

        Span<double> b = [5, 6, 7, 8];
        impl1.Tensor<double> B = new impl1.Tensor<double>(b, 1, 4);

        Span<double> c = stackalloc double[2 * 2 * 2 * 2];
        impl1.Tensor<double> C = impl1.Tensor<double>.OuterProduct(A, B, c);

        Console.WriteLine(A[2, 0]);
        Console.WriteLine(B[0, 3]);
        Console.WriteLine(C[2, 0, 0, 2]);
    }

    static void AlternateImplementation()
    {
        Span<double> a = [1, 2, 3, 4];
        impl2.Tensor<double> A = new impl2.Tensor<double>(a, ['i'], [], [4]);

        Span<double> b = [5, 6, 7, 8];
        impl2.Tensor<double> B = new impl2.Tensor<double>(b, [], ['j'], [4]);

        Span<double> c = stackalloc double[2 * 2 * 2 * 2];
        impl2.Tensor<double> C = impl2.Tensor<double>.OuterProduct(A, B, c);

        Console.WriteLine(A[('i', 2)]);
        Console.WriteLine(B[('j', 3)]);
        Console.WriteLine(C[('i', 2), ('j', 2)]);
    }

    static void KroneckerProduct()
    {
        Span<double> a = [1, -4, 7, -2, 3, 3];
        impl1.Tensor<double> A = new impl1.Tensor<double>(a, 2, 3);

        Span<double> b = [8, -9, -6, 5, 1, -3, -4, 7, 2, 8, -8, -3, 1, 2, -5, -1];
        impl1.Tensor<double> B = new impl1.Tensor<double>(b, 4, 4);

        Span<double> c = stackalloc double[2 * 3 * 4 * 4];
        impl1.Tensor<double> C = impl1.Tensor<double>.KroneckerProduct(A, B, c);

        Console.WriteLine(A[1, 0]);
        Console.WriteLine(B[0, 3]);
        Console.WriteLine(C[4, 5]);
        Console.WriteLine(String.Join(", ", c.ToArray()));
    }

    static void Add()
    {
        Span<double> a = [1, 2, 3, 4];
        impl1.Tensor<double> A = new impl1.Tensor<double>(a, 1, 4);

        Span<double> b = [5, 6, 7, 8];
        impl1.Tensor<double> B = new impl1.Tensor<double>(b, 1, 4);

        Span<double> dest = stackalloc double[2 * 2];
        impl1.Tensor<double> C = impl1.Tensor<double>.Add(A, 1.0, B, 1.5, dest);

        Console.WriteLine(String.Join(", ", dest.ToArray()));
    }
}
