/*
 * This file is part of cgreenlet. CGreenlet is free software available
 * under the terms of the MIT license. Consult the file LICENSE that was
 * shipped together with this source file for the exact licensing terms.
 *
 * Copyright (c) 2012 by the cgreenlet authors. See the file AUTHORS for a
 * full list.
 */

#if defined(__WIN32__) || defined(__APPLE__)
# define cdecl(s) _##s
#else
# define cdecl(s) s
#endif

.text

.globl cdecl(_greenlet_savecontext)
.globl cdecl(_greenlet_switchcontext)
.globl cdecl(_greenlet_newstack)

#if defined(__x86_64__)

cdecl(_greenlet_savecontext):
    .cfi_startproc
    .cfi_def_cfa %rsp,8
    movq (%rsp), %rax
    movq %rax, (%rdi)
    movq %rsp, %rax
    addq $0x8, %rax
    movq %rax, 0x8(%rdi)
    movq %rbp, 0x10(%rdi)
    movq %rbx, 0x18(%rdi)
    movq %r12, 0x20(%rdi)
    movq %r13, 0x28(%rdi)
    movq %r14, 0x30(%rdi)
    movq %r15, 0x38(%rdi)
    movq $0x0, %rax
    ret
    .cfi_endproc

cdecl(_greenlet_switchcontext):
    .cfi_startproc
    .cfi_def_cfa %rsp,0
    movq 0x8(%rdi), %rsp
    movq 0x10(%rdi), %rbp
    movq 0x18(%rdi), %rbx
    movq 0x20(%rdi), %r12
    movq 0x28(%rdi), %r13
    movq 0x30(%rdi), %r14
    movq 0x38(%rdi), %r15
    testq %rsi, %rsi
    jnz 1f
    movq $0x1, %rax
    jmpq *(%rdi)
1:
    pushq (%rdi)
    .cfi_def_cfa %rsp,8
    movq %rdx, %rdi
    callq *%rsi
    movq $0x1, %rax
    ret
    .cfi_endproc

cdecl(_greenlet_newstack):
    .cfi_startproc
    .cfi_def_cfa %rsp,8
    movq %rdi, %rsp
    movq %rdx, %rdi
    pushq $0x0
    jmpq *%rsi
    .cfi_endproc
 
#elif defined(__i386__)

cdecl(_greenlet_savecontext):
    .cfi_startproc
    .cfi_def_cfa %esp,8
    movl %edi, %eax
    movl 0x4(%esp), %edi
    movl (%esp), %ecx
    movl %ecx, (%edi)
    movl %esp, %ecx
    addl $0x4, %ecx
    movl %ecx, 0x4(%edi)
    movl %ebp, 0x8(%edi)
    movl %ebx, 0xc(%edi)
    movl %esi, 0x10(%edi)
    movl %eax, 0x14(%edi)
    movl %eax, %edi
    movl $0x0, %eax
    ret
    .cfi_endproc

cdecl(_greenlet_switchcontext):
    .cfi_startproc
    .cfi_def_cfa %esp,0
    movl 0x4(%esp), %edi
    movl 0x8(%esp), %eax
    movl 0xc(%esp), %ecx
    movl (%edi), %edx
    movl 0x4(%edi), %esp
    movl 0x8(%edi), %ebp
    movl 0xc(%edi), %ebx
    movl 0x10(%edi), %esi
    movl 0x14(%edi), %edi
    testl %eax, %eax
    jnz .Linject
    movl $0x1, %eax
    jmpl *%edx
.Linject:
    pushl %edx
    .cfi_def_cfa %esp,4
    pushl %ecx
    .cfi_def_cfa %esp,8
    calll *%eax
    popl %ecx
    .cfi_def_cfa %esp,4
    movl $0x1, %eax
    ret
    .cfi_endproc

cdecl(_greenlet_newstack):
    .cfi_startproc
    .cfi_def_cfa %esp,4
    movl %esp, %edi
    movl 0x4(%edi), %esp
    movl 0x8(%edi), %eax
    movl 0xc(%edi), %ecx
    pushl %ecx
    pushl $0x0
    jmpl *%eax
    .cfi_endproc

#endif
