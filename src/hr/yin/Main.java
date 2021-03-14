package hr.yin;

import java.util.*;

public class Main {

    public static void main(String[] args) {
    }

    /**
     * 数组中重复数字
     * 0.两重for
     * 1.HashSet(HashMap)判断当前数字是否出现过 (相比于0：空间换时间
     * 2.bucket数组记录每个数字是否出现过 (相比于1：数组代替哈希表
     * 3.数字和数组下标一致 (相比于2：减小空间复杂度
     */
    public int findRepeatNumber(int[] nums) {
        // 从第一个元素开始遍历数组
        for (int i = 0; i < nums.length; i++) {
            // 当前数字与下标不一致时，将数字交换到正确下标
            while (nums[i] != i) {
                // 当前数字待交换去的位置有正确的数字，找到重复元素
                if (nums[nums[i]] == nums[i]) {
                    return nums[i];
                }

                int tmp = nums[nums[i]];
                nums[nums[i]] = nums[i];
                nums[i] = tmp;
            }
        }

        // 没有重复元素
        return -1;
    }

    /**
     * 排序二维数组中查找
     * 左上角为最小值，右下角为最大值
     */
    public boolean findNumberIn2DArray(int[][] matrix, int target) {
        // 从右上角元素比较，每次比较可排除一行/列
        int row = 0;
        int column = matrix[0].length - 1;

        while (row < matrix.length && column >= 0) {
            if (matrix[row][column] == target) {
                return true;
            } else if (target > matrix[row][column]) {
                row++;
            } else {
                column--;
            }
        }

        // 整个数组未找到target
        return false;
    }

    /**
     * 替换空格
     * 1.String#replaceAll() 性能不好
     * 2.StringBuilder拼接   本质是：char[]+扩容（System.arraycopy()）
     * 3.定义静态数组：大小刚好（遍历两次字符串，损失时间）、大小为字符串3倍（损失空间）
     */
    public String replaceSpace(String s) {
        if (s == null) {
            return null;
        }

        StringBuilder sb = new StringBuilder();

        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == ' ') {
                sb.append("%20");
            } else {
                sb.append(s.charAt(i));
            }
        }

        return sb.toString();
    }

    public static class ListNode {
        int val;
        ListNode next;

        ListNode(int x) {
            val = x;
        }

        ListNode(int x, ListNode next) {
            val = x;
            this.next = next;
        }
    }

    /**
     * 从尾到头打印单链表
     * 必须从头到尾遍历链表
     * 1.可用栈(Stack extends Vector)存储遍历的每个值（多用一个栈的空间）
     * 2.访问一次链表记录节点数、再次访问链表存储节点值（多了遍历一次的时间）
     */
    public int[] reversePrint(ListNode head) {
        ListNode p = head;
        int count = 0;
        while (p != null) {
            count++;
            p = p.next;
        }

        int[] result = new int[count];
        while (head != null) {
            result[--count] = head.val;
            head = head.next;
        }

        return result;
    }

    public static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }

    /**
     * 前序和中序重建二叉树。不含重复值
     * 1.直接在中序遍历中查找头结点位置
     * 2.HashMap存储中序遍历值及其下标（提高了查找速度，但增加了空间的使用
     */
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        if (preorder == null || preorder.length == 0 || inorder == null || inorder.length == 0) {
            return null;
        }

        // HashMap存储中序遍历元素及其下标，方便查找
        Map<Integer, Integer> inMap = new HashMap<>();
        for (int i = 0; i < inorder.length; i++) {
            inMap.put(inorder[i], i);
        }

        return buildTree(preorder, 0, preorder.length - 1, inorder, 0, inorder.length - 1, inMap);
    }

    private TreeNode buildTree(int[] preorder, int preStart, int preEnd,
                               int[] inorder, int inStart, int inEnd,
                               Map<Integer, Integer> inMap) {
        // 边界
        if (preStart > preEnd || inStart > inEnd) {
            return null;
        }

        // 当前确定的头结点
        TreeNode head = new TreeNode(preorder[preStart]);

        // 头结点在中序的下标
        int headPosInOrder = inMap.get(head.val);
        // 左孩子节点数
        int leftChildNum = headPosInOrder - inStart;
        head.left = buildTree(preorder, preStart + 1, preStart + leftChildNum, inorder, inStart, headPosInOrder - 1, inMap);
        head.right = buildTree(preorder, preStart + leftChildNum + 1, preEnd, inorder, headPosInOrder + 1, inEnd, inMap);
        return head;
    }

    /**
     * 两个栈实现队列
     */
    class CQueue {

        private final Stack<Integer> inputStack;
        private final Stack<Integer> outputStack;

        public CQueue() {
            inputStack = new Stack<>();
            outputStack = new Stack<>();
        }

        public void appendTail(int value) {
            inputStack.push(value);
        }

        public int deleteHead() {
            if (!outputStack.isEmpty()) {
                return outputStack.pop();
            }

            if (inputStack.isEmpty()) {
                return -1;
            } else {
                while (!inputStack.isEmpty()) {
                    outputStack.push(inputStack.pop());
                }
                return outputStack.pop();
            }
        }

    }

    /**
     * 斐波那契数列
     * 0 1 1 2 3 5 ...
     */
    public int fib(int n) {
        if (n == 0 || n == 1) {
            return n;
        }

        int firstNumber = 0;
        int secondNumber = 1;
        int tmp;
        while (n >= 2) {
            tmp = (firstNumber + secondNumber) % 1000000007;
            firstNumber = secondNumber;
            secondNumber = tmp;
            --n;
        }
        return secondNumber;
    }

    /**
     * 青蛙跳台阶
     * f(n) = f(n-1) + f(n-2)
     * f(0) = 1   f(1) = 1   f(2) = 2   f(3) = 3
     */
    public int numWays(int n) {
        if (n == 0 || n == 1) {
            return 1;
        }

        int firstNum = 1;
        int secondNum = 1;
        int tmp;
        while (n >= 2) {
            tmp = (firstNum + secondNum) % 1000000007;
            firstNum = secondNum;
            secondNum = tmp;
            --n;
        }

        return secondNum;
    }

    /*
        二分查找：
            左、中、右。（以中间值作为基准,进行大、小、相等的比较）
            每次查找会排除掉一些元素，且数组元素特性保持不变
            注意分析数组元素个数为1、2的情况
     */
    /**
     * 旋转数组的最小值
     *
     * 数组情况：没旋转---递增序列(最小值在左边)    旋转了---最小值靠左边、最小值靠右边
     * 特殊值：第一个元素、最后个元素
     *
     * 中间值和边界值相等->不能判断最小值在左边或右边->直接忽略边界值
     *
     * 不能分类的情况，可以用特殊条件来过滤。
     * 以第一个元素作为待比较值，当numbers[mid] > numbers[left]时：1.没旋转，最小值在左边，为第一个元素。2.旋转了，最小值在右边
     * 此时可以判断：如果numbers[left] < number[right]，那么为1，直接返回最小值numbers[left]
     */
    public int minArray(int[] numbers) {
        if (numbers == null || numbers.length == 0) {
            return -1;
        }

        int left = 0;
        int right = numbers.length - 1;
        int mid;
        while (left < right) {
            mid = (left + right) / 2; // mid = left + (right - left) / 2;
            if (numbers[mid] > numbers[right]) {
                left = mid + 1;
            } else if (numbers[mid] < numbers[right]) {
                right = mid;
            } else {
                right = right - 1;
            }
        }

        return numbers[left];
    }
    /*public int minArray(int[] numbers) {
        if (numbers == null || numbers.length == 0) {
            return -1;
        }

        int left = 0;
        int right = numbers.length - 1;
        int mid;
        while (right - left > 1) {
            mid = (left + right) / 2;
            if (numbers[mid] > numbers[left]) {
                if(numbers[left] < numbers[right]) {
                    return numbers[left];
                }
                left = mid + 1;
            } else if (numbers[mid] < numbers[left]) {
                right = mid;
            } else {
                left =left + 1;
            }
        }
        return numbers[left] > numbers[right] ? numbers[right] : numbers[left];
    }*/

    /**
     * 矩阵中的路径
     *
     * DFS 回溯 递归
     */
    public boolean exist(char[][] board, String word) {
        if (board == null || board.length == 0 || board[0] == null || board[0].length == 0) {
            return false;
        }
        if (word == null || word.length() == 0) {
            return false;
        }

        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {
                if (checkDFS(board, i, j, word, 0)) {
                    return true;
                }
            }
        }

        return false;
    }
    /**
     * @param board 矩阵
     * @param i 当前行
     * @param j 当前列
     * @param word 路径
     * @param index 当前要比较字符的下标
     */
    private boolean checkDFS(char[][] board, int i, int j, String word, int index) {
        if (index == word.length()) {
            return true;
        }

        if (i < 0 || i >= board.length ||
            j < 0 || j >= board[0].length ||
            board[i][j] != word.charAt(index)) {
            return false;
        }

        board[i][j] = '\0';
        boolean result =
            checkDFS(board, i + 1, j, word, index + 1) ||
            checkDFS(board, i - 1, j, word, index + 1) ||
            checkDFS(board, i, j + 1, word, index + 1) ||
            checkDFS(board, i, j - 1, word, index + 1);
        board[i][j] = word.charAt(index);

        return result;
    }

    /**
     * 机器人的运动范围
     *
     * DFS 递归
     * BFS 队列
     */
    public int movingCount(int m, int n, int k) {
        if (m <= 0 || n <= 0 || k < 0) {
            return 0;
        }

        boolean[][] visited = new boolean[m][n];
        return movingDFS(0, 0, k, visited);
    }
    /**
     *
     * @param i 当前行号
     * @param j 当前列号
     * @param k 下标数位之和的限制
     * @param visited 标记被访问过的位置
     * @return 当前位置继续移动，总共可以移动的位置数量
     */
    private int movingDFS(int i, int j, int k, boolean[][] visited) {
        if (i >= visited.length || i < 0 ||
            j >= visited[0].length || j < 0 ||
            visited[i][j] || getNumSum(i) + getNumSum(j) > k) {
            return 0;
        }

        visited[i][j] = true;
        return 1 +
            movingDFS(i + 1, j, k, visited) +
            movingDFS(i, j + 1, k, visited);
    }
    /** 获取一个数的数位之和 */
    private int getNumSum(int num) {
        int sum = 0;

        while (num != 0) {
            sum += num % 10;
            num /= 10;
        }

        return sum;
    }

    /**
     * 剪绳子
     *
     * 1.动态规划
     * 2.数学知识：算术几何均值不等值->求导取极大值->e为驻点。
     *           <p>最优：3   次优：2   特殊：3*1<2*2</p>
     */
    public int cuttingRope(int n) {
        if (n < 2) {
            return 0;
        }

        // 长度为下标的绳子剪了的乘积和原本长度的最大值
        int[] max = new int[n];
        max[1] = 1;

        // 长度为n-1的绳子所能取到的剪了的乘积和原本长度的最大值
        for (int i = 2; i <= n - 1; i++) {
            max[i] = i;
            // 对折一半
            for (int j = 1; j <= i / 2; j++) {
                max[i] = Math.max(max[i], max[j] * max[i - j]);
            }
        }

        // 长度为n的绳子，剪了的乘积的最大值
        int result = max[1] * max[n - 1];
        for (int i = 2; i <= n - 1; i++) {
            result = Math.max(result, max[i] * max[n - i]);
        }

        return result;
    }
    /*public int cuttingRope(int n) {
        if (n < 2) {
            return 0;
        }
        if (n == 2) {
            return 1;
        }
        if (n == 3) {
            return 2;
        }

        // n >= 4
        int amountOf3 = n / 3;
        int remainder = n % 3;
        int result = (int) Math.pow(3, amountOf3);
        if (remainder == 2) {
            result *= 2;
        }
        if (remainder == 1) {
            result = result / 3 * 4;
        }
        // 余数为0，不用再做再做操作

        return result;
    }*/

    /**
     * 剪绳子 - 大数问题
     *
     * 大数问题：注意int最大值为2开头10的9次方，两个int相乘不会超过long，多用long
     *
     * 快速幂：a^b = (a^2)^(b/2)   or   a*(a^2)^(b/2)
     *
     * (a * b) % p = (a % p * b % p) % p
     */
    public int cuttingRope2(int n) {
        if (n < 2) {
            return 0;
        }
        if (n == 2) {
            return 1;
        }
        if (n == 3) {
            return 2;
        }

        // 3的个数
        int amountOf3 = n / 3;
        // 除3后的余数
        int remainder = n % 3;

        int modNumber = 1000000007;

        // 快速幂
        long base = 3;
        int power = amountOf3 - 1;
        long result = 1;
        // power等于1，result乘最后的base。power等于0，结束循环
        while(power > 0) {
            if ((power & 1) == 1) {
                result = result * base % modNumber;
            }
            base = base * base % modNumber;
            power >>>= 1;
        }

        if (remainder == 2) {
            result = result * 2 * 3 % modNumber;
        }
        if (remainder == 1) {
            result = result * 4 % modNumber;
        }
        if (remainder == 0) {
            result = result * 3 % modNumber;
        }

        return (int) result;
    }

    /**
     * 二进制中1的个数
     */
    public int hammingWeight(int n) {
        int count = 0;

        while (n != 0) {
            count++;
            // 最低位1置0
            n = n & (n - 1);
        }

        return count;
    }

    /**
     * 数值的整数次方。忽略大数问题。
     *
     * 1.边界问题
     * 2.快速幂
     * 3.注意：最小值的相反数/绝对值会超过最大值.
     * {@link Integer#MIN_VALUE}、{@link Integer#MAX_VALUE}.
     * -Integer.MIN_VALUE 等于 Math.abs(Integer.MIN_VALUE) 等于 Integer.MIN_VALUE
     */
    public double power(double base, int exponent) throws ArithmeticException {
        if (base == 0 && exponent < 0) {
            throw new ArithmeticException("除0操作");
        }

        // 避免出现int的最小值取绝对值导致的数值溢出问题
        double result = fastPower(base, Math.abs((long) exponent));

        if (exponent < 0) {
            result = 1/ result;
        }

        return result;
    }
    /**
     * 快速幂。指数为非负整数
     */
    private double fastPower(double base, long exponent) {
        double result = 1;

        while (exponent > 0) {
            if ((exponent & 1) == 1) {
                result *= base;
            }
            base *= base;
            exponent >>= 1;
        }

        return result;
    }

    /**
     * 打印1到最大的n位数
     *
     * 1.用int long：有类型存在最大的限制
     * 2.全排列：dfs
     * 3.String模拟加法、BigInteger
     *
     * 注：时间复杂度一样
     */
    public void printNumbers(int n) {
        dfs(new StringBuilder(), n);
    }
    /**
     * 全排列 dfs
     * @param number 当前已确定位置的数。如第n位 第n-1位
     * @param position 待确定的位置。如第n-2位
     */
    private void dfs(StringBuilder number, int position) {
        // 待确定的位置为0，说明整个数拼接完成
        if (position == 0) {
            if (number.length() != 0) {
                System.out.print(number.toString() + " ");
            }
            return;
        }

        for (int i = 0; i <= 9; i++) {
            if (number.length() == 0 && i == 0) {
                dfs(number, position - 1);
            } else {
                number.append(i);
                dfs(number, position - 1);
                number.deleteCharAt(number.length() - 1);
            }
        }
    }

    /**
     * 删除单链表节点。参数：head节点、待删除值
     *
     * 对于类似的问题需要考虑的特殊情况：0节点、1节点、head节点、tail节点
     *
     * 扩展：在O(1)时间删除单链表存在的节点。参数：head节点、待删除节点
     * 解法：待删除节点的后继内容复制到待删除节点，删除后继。注意尾结点和只有一个节点
     */
    public ListNode deleteNode(ListNode head, int val) {
        // 定义head节点的前驱，统一删除操作
        ListNode preHead = new ListNode(0);
        preHead.next = head;

        // 遍历单链表的引用
        ListNode p = preHead;
        // 遍历单链表
        while (p.next != null && p.next.val != val) {
            p = p.next;
        }

        // 找到了待删除的节点，删除
        if (p.next != null) {
            p.next = p.next.next;
        }

        return preHead.next;
    }

    /**
     * 正则表达式匹配。'*'、'.'和a-z
     *
     * 1.dp。
     *  前面的已确定，确定当前
     *    广义相同(相同/regex为'.')   取决于dp[i - 1][j - 1]
     *    广义不同(与上相反)   1.regex当前为'*'：1.1regex前一个值等于str当前值：取决于dp[i - 1][j] || dp[i][j - 2]; 1.2不等：false
     *                      2.regex当前不为‘*’：false
     *
     * 2.dfs。
     * 前面的已匹配，比较当前
     *  广义相同(相同/regex为'.')   1.regex下一个为'*'：reg移动两步/str移动一步  2.regex下一个不为'*'：均移动一步
     *  广义不同(与上相反)   1.regex下一个为"*"：regex移动两步   2.regex下一个不为"*"：返回false
     *  注意：有相同子问题就记录已知结果。结束比较是regex不为空的特殊情况。
     */
    // dfs
    public boolean isMatchDFS(String str, String regex) {
        if (str == null || regex == null) {
            return false;
        }

        // 记录已经确定的不匹配子序列。noMatch[i][j]为true代表str的i及i以后的子序列和regex的j及j以后的子序列不匹配，为false才需要dfs。
        boolean[][] noMatch = new boolean[str.length() + 1][regex.length() + 1];

        return dfs(str, 0, regex, 0, noMatch);
    }
    // dfs判断子序列是否匹配
    private boolean dfs(String str, int p1, String regex, int p2, boolean[][] noMatch) {
        // str或regex遍历完成结束递归
        if (p1 == str.length() || p2 == regex.length()) {
            // 均遍历完成，匹配
            if (p1 == str.length() && p2 == regex.length()) {
                return true;
            }
            // str遍历未完成，不匹配
            if (p1 != str.length()) {
                return false;
            }
            // regex遍历未完成，需判断
            // p2 != regex.length()
            while (p2 != regex.length()) {
                if (regex.charAt(p2) == '*') {
                    p2++;
                } else {
                    if (p2 != regex.length() - 1 && regex.charAt(p2 + 1) == '*') {
                        p2 += 2;
                    } else {
                        return false;
                    }
                }
            }
            return true;
        }

        // 广义相同
        if (str.charAt(p1) == regex.charAt(p2) || regex.charAt(p2) == '.') {
            // regex下一个为'*"
            if (p2 != regex.length() - 1 && regex.charAt(p2 + 1) == '*') {
                // 忽略regex当前值、匹配regex当前值
                if ((!noMatch[p1][p2 + 2] && dfs(str, p1, regex, p2 + 2, noMatch))
                        || (!noMatch[p1 + 1][p2] && dfs(str, p1 + 1, regex, p2, noMatch))) {
                    return true;
                }
            } else { // regex下一个不为'*"
                // regex和str均移动一步
                if (!noMatch[p1 + 1][p2 + 1] && dfs(str, p1 + 1, regex, p2 + 1, noMatch)) {
                    return true;
                }
            }
        }
        // 广义不同
        // regex下一个为'*",则可忽略这次的不同，regex移动两步
        if (p2 != regex.length() - 1 && regex.charAt(p2 + 1) == '*'
                && !noMatch[p1][p2 + 2] && dfs(str, p1, regex, p2 + 2, noMatch)) {
            return true;
        } else {
            noMatch[p1][p2] = true;
            return false;
        }
    }
    // dp
    public boolean isMatchDP(String str, String regex) {
        if (str == null || regex == null) {
            return false;
        }

        // 记录已知结果。dp[i][j]表示str的前i个和regex的前j个是否匹配。注意下标。
        boolean[][] dp = new boolean[str.length() + 1][regex.length() + 1];
        // d[i][0](i > 0)都为false
        dp[0][0] = true;

        // 确定str的每个子序列是否匹配regex的每个子序列。注意str为空字符串且regex不为空字符串。
        for (int i = 0; i <= str.length(); i++) {
            for (int j = 1; j <= regex.length(); j++) {
                // 当前值相等
                if (i > 0 && (regex.charAt(j - 1) == '.' || str.charAt(i - 1) == regex.charAt(j - 1))) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else { // 当前值不等
                    // regex当前值为*
                    if (regex.charAt(j - 1) == '*') {
                        // 1.忽略
                        if (j >= 2) {
                            dp[i][j] |= dp[i][j - 2];
                        }
                        // 2.匹配
                        if (i > 0 && j >= 2 && (str.charAt(i - 1) == regex.charAt(j - 2) || regex.charAt(j - 2) == '.')) {
                            dp[i][j] |= dp[i - 1][j];
                        }

                    } // else false
                }
            }
        }

        return dp[str.length()][regex.length()];
    }

    /**
     * 数组中使奇数位于偶数前面
     * <br>类似于快速排序算法
     */
    public int[] exchange(int[] nums) {
        if (nums == null) {
            return null;
        }

        // 从头往后遍历数组找偶数
        int start = 0;
        // 从尾往前遍历数组找奇数
        int end = nums.length - 1;
        // 不可能相等
        while (start < end) {
            // start为奇数则继续往后查找
            while (start < nums.length && (nums[start] & 1) == 1) {
                start++;
            }
            // end为偶数则继续往前查找
            while (end >= 0 && (nums[end] & 1) == 0) {
                end--;
            }

            if (start < end) {
                int tmp = nums[start];
                nums[start] = nums[end];
                nums[end] = tmp;
            }
        }

        return nums;
    }

    /**
     * 链表中倒数第k个节点
     */
    public ListNode getKthFromEnd(ListNode head, int k) {
        if (head == null || k <= 0) {
            return null;
        }

        // 先走k-1步的对象引用
        ListNode p1 = head;
        // 待返回结果的对象引用
        ListNode p2 = head;

        for (int i = 0; i < k - 1; i++) {
            // 链表长度小于k
            if (p1 == null) {
                return null;
            }
            p1 = p1.next;
        }

        // p1走到最后个结点结束
        while (p1.next != null) {
            p1 = p1.next;
            p2 = p2.next;
        }

        return p2;
    }

    /**
     * 反转链表
     */
    public ListNode reverseList(ListNode head) {
        if (head == null) {
            return null;
        }

        // 反转后的链表头结点
        ListNode result = null;
        // 当前反转结点
        ListNode current = head;
        // 暂存当前反转结点的下一个结点
        ListNode next;
        while (current != null) {
            next = current.next;

            // 反转
            current.next = result;

            result = current;
            current = next;
        }

        return result;
    }

    /**
     * 合并两个有序链表
     * 1 3 5
     * 2 4 6
     */
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        // 合并后链表的头结点
        ListNode headNode = new ListNode(0, null);
        // 当前已合并完成链表的节点
        ListNode current = headNode;

        // l1 l2分别记录两个链表当前待比较节点
        while (l1 != null && l2 != null) {
            if (l1.val <= l2.val) {
                current.next = l1;
                l1 = l1.next;
            } else {
                current.next = l2;
                l2 = l2.next;
            }
            current = current.next;
        }

        if (l1 != null) {
            current.next = l1;
        }
        if (l2 != null) {
            current.next = l2;
        }

        return headNode.next;
    }

    /**
     * 子树问题
     *
     * <br> 树相关的问题 -> 递归、迭代
     * <br> 树的遍历(先序、中序、后序、层次) -> 递归、迭代。   注：递归的先序比迭代的层次快非常多。
     */
    public boolean isSubStructure(TreeNode A, TreeNode B) {
        // 约定空树不是任意一个树的子结构
        if (B == null) {
            return false;
        }

        // 父结构为空
        if (A == null) {
            return false;
        }

        // 递归先序遍历父结构
        return sub(A, B) || isSubStructure(A.left, B) || isSubStructure(A.right, B);
    }
    /**
     * 判断是不是子树
     */
    public boolean sub(TreeNode tree, TreeNode subTree) {
        // 子树节点比较完成，再没节点了
        if (subTree == null) {
            return true;
        }

        if (tree == null) {
            return false;
        }

        return tree.val == subTree.val && sub(tree.left, subTree.left) && sub(tree.right, subTree.right);
    }

    /**
     * 二叉树的镜像
     */
    public TreeNode mirrorTree(TreeNode root) {
        return exchangeChildren(root);
    }
    /**
     * 交换节点的两个子节点
     */
    private TreeNode exchangeChildren(TreeNode node) {
        if (node == null) {
            return null;
        }
        if (node.left == null && node.right == null) {
            return node;
        }

        TreeNode rightChild = node.right;
        node.right = exchangeChildren(node.left);
        node.left = exchangeChildren(rightChild);
        return node;
    }

    /**
     * 对称的二叉树
     */
    public boolean isSymmetric(TreeNode root) {
        if (root == null) {
            return true;
        }

        return isChildrenSymmetric(root.left, root.right);
    }
    private boolean isChildrenSymmetric(TreeNode child1, TreeNode child2) {
        if (child1 == null && child2 == null) {
            return true;
        }
        if (child1 == null || child2 == null) {
            return false;
        }

        return child1.val == child2.val
                && isChildrenSymmetric(child1.left, child2.right)
                && isChildrenSymmetric(child1.right, child2.left);

    }

    /**
     * 顺时针打印矩阵
     */
    public int[] spiralOrder(int[][] matrix) {
        if (matrix == null || matrix.length == 0
                || matrix[0] == null || matrix[0].length == 0) {
            return new int[0];
        }

        // 四个方向的边界，可以取
        int left = 0, right = matrix[0].length -1, top = 0, bottom = matrix.length - 1;

        // 遍历结果数组
        int[] result = new int[(right + 1) * (bottom + 1)];
        int n = 0;

        while(true) {
            // 右
            for (int i = left; i <= right; i++) {
                result[n++] = matrix[top][i];
            }
            // 上边界遍历完
            if (++top > bottom) {
                break;
            }

            // 下
            for (int i = top; i <= bottom; i++) {
                result[n++] = matrix[i][right];
            }
            // 右边界遍历完
            if (--right < left) {
                break;
            }

            // 左
            for (int i = right; i >= left; i--) {
                result[n++] = matrix[bottom][i];
            }
            // 下边界遍历完
            if (--bottom < top) {
                break;
            }

            // 上
            for (int i = bottom; i >= top; i--) {
                result[n++] = matrix[i][left];
            }
            // 左边界遍历完
            if (++left > right) {
                break;
            }
        }

        return result;
    }
    /**
     * 这是不好的方式，上面的方式做了如下改进
     *
     * 改进1(空间复杂度)：不用visited数组，改用四个边界值。
     * 改进2(虽然时间复杂度一样，但基本语句更花时间)：不用每次判断方向，一次遍历完一个方向的值。
     */
    public int[] spiralOrder2(int[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            return new int[0];
        }

        // 总的行列数
        int row = matrix.length;
        int col = matrix[0].length;

        // 标记是否访问过该点
        boolean[][] visited = new boolean[row][col];

        // 遍历结果数组
        int[] result = new int[row * col];
        // 当前移动方向，0 1 2 3 右 下 左 上
        int direction = 0;
        // 当前行列值
        int i = 0;
        int j = -1;
        for (int n = 0; n < result.length; n++) {
            switch (direction) {
                case 0:
                    // 可以继续往该方向遍历
                    if (j + 1 < col && !visited[i][j + 1]) {
                        result[n] = matrix[i][++j];
                        visited[i][j] = true;
                        break;
                    }
                    // 此方向已遍历完成，换个方向
                    direction = 1;
                case 1:
                    // 可以继续往该方向遍历
                    if (i + 1 < row && !visited[i + 1][j]) {
                        result[n] = matrix[++i][j];
                        visited[i][j] = true;
                        break;
                    }
                    // 此方向已遍历完成，换个方向
                    direction = 2;
                case 2:
                    // 可以继续往该方向遍历
                    if (j - 1 >= 0 && !visited[i][j - 1]) {
                        result[n] = matrix[i][--j];
                        visited[i][j] = true;
                        break;
                    }
                    // 此方向已遍历完成，换个方向
                    direction = 3;
                case 3:
                    // 可以继续往该方向遍历
                    if (i - 1 > 0 && !visited[i - 1][j]) {
                        result[n] = matrix[--i][j];
                        visited[i][j] = true;
                        break;
                    }
                    // 此方向已遍历完成，换个方向
                    direction = 0;
                    n--;
            }
        }

        return result;
    }

    /**
     * 包含min方法的栈(时间复杂度O(1))
     *
     * 1. push时多存储一个当前的最小值
     * 2. push时存储非严格降序元素
     *
     * 注：Integer使用equal比较，不用==
     */
    public void min() {}

    /**
     * 栈的压入、弹出序列
     */
    public boolean validateStackSequences(int[] pushed, int[] popped) {
        if (pushed == null || popped == null || pushed.length != popped.length) {
            return false;
        }

        // pushed待比较元素下标，i前面的都是未弹出的
        int i = 0;
        // popped的遍历下标
        int j = 0;

        for (int e : pushed) {
            pushed[i] = e;
            while (i >= 0 && pushed[i] == popped[j]) {
                i--;
                j++;
            }
            i++;
        }

        return i == 0;
    }
    public boolean validateStackSequences2(int[] pushed, int[] popped) {
        if (pushed == null || popped == null || pushed.length != popped.length) {
            return false;
        }

        Stack<Integer> stack = new Stack<>();
        int i = 0;

        for (int e : pushed) {
            stack.push(e);
            while (!stack.isEmpty() && stack.peek() == popped[i]) {
                stack.pop();
                i++;
            }
        }

        return stack.isEmpty();
    }
    public boolean validateStackSequences3(int[] pushed, int[] popped) {
        if (pushed == null || popped == null || pushed.length != popped.length) {
            return false;
        }

        // 保存暂时不匹配的数据
        Stack<Integer> stack = new Stack<>();
        int pushIndex = 0, popIndex = 0;

        // 取pop数组当前值
        // 1.和push数组比较
        // 2.和stack比较
        // 3.执行入栈或返回false
        while(popIndex < popped.length) {
            if (pushIndex < pushed.length && popped[popIndex] == pushed[pushIndex]) {
                popIndex++;
                pushIndex++;
                continue;
            }

            if (!stack.isEmpty() && popped[popIndex] == stack.peek()) {
                stack.pop();
                popIndex++;
                continue;
            }

            if (pushIndex < pushed.length) {
                stack.push(pushed[pushIndex]);
                pushIndex++;
            } else {
                return false;
            }
        }

        return stack.isEmpty();
    }

    /**
     * 二叉树层次遍历
     */
    public int[] levelOrder(TreeNode root) {
        if (root == null) {
            return new int[0];
        }

        Queue<TreeNode> queue = new LinkedList<>();
        List<Integer> levelSeq = new ArrayList<>();
        queue.offer(root);

        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            levelSeq.add(node.val);
            if (node.left != null) {
                queue.offer(node.left);
            }
            if (node.right != null) {
                queue.offer(node.right);
            }
        }

        int size = levelSeq.size();
        int[] result = new int[size];
        for (int i = 0; i < size; i++) {
            result[i] = levelSeq.get(i);
        }
        return result;
    }

    /**
     * 二叉树层次遍历。按层存储。
     */
    public List<List<Integer>> levelOrder2(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }

        Queue<TreeNode> queue = new LinkedList<>();
        List<List<Integer>> result = new ArrayList<>();
        List<Integer> levelSequence = new ArrayList<>();
        queue.offer(root);

        while (!queue.isEmpty()) {
            for (int i = queue.size(); i > 0; i--) {
                TreeNode node = queue.poll();
                levelSequence.add(node.val);
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            }

            result.add(levelSequence);
            levelSequence = new ArrayList<>();
        }

        return result;
    }
}
