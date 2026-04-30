
using System.Numerics.Tensors;

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
        Tensor<double> A = new Tensor<double>(a, 2, 2);

        Span<double> b = [1, 1, 1, -1];
        Tensor<double> B = new Tensor<double>(b, 2, 2);

        Span<double> c = stackalloc double[2 * 2 * 2 * 2];
        Tensor<double> C = Tensor<double>.OuterProduct(A, B, c);

        Console.WriteLine(C[0, 0, 1, 1]);
    }

    static void OuterProduct2()
    {
        Span<double> a = [1, 2, 3, 4];
        Tensor<double> A = new Tensor<double>(a, 4, 1);

        Span<double> b = [5, 6, 7, 8];
        Tensor<double> B = new Tensor<double>(b, 1, 4);

        Span<double> c = stackalloc double[2 * 2 * 2 * 2];
        Tensor<double> C = Tensor<double>.OuterProduct(A, B, c);

        Console.WriteLine(A[2, 0]);
        Console.WriteLine(B[0, 3]);
        Console.WriteLine(C[2, 0, 0, 2]);
    }
}
