	.global main_func
	.global _start
	.intel_syntax noprefix
	.text
my_strcmp:
	xor	ecx, ecx
.L3:
	movzx	edx, BYTE PTR [rsi+rcx]
	test	al, al
	je	.L2
	cmp	al, dl
	je	.L3
.L2:
	sub	eax, edx
	.size	my_strcmp, .-my_strcmp
write_str:
	mov	rsi, rdi
.L10:
	cmp	BYTE PTR [rsi+rdx], 0
	je	.L12
	inc	rdx
	jmp	.L10
.L12:
	mov	eax, 1
	mov	edi, eax
	syscall
	ret
	.size	write_str, .-write_str
write_dec_padded:
	sub	rsp, 32
	mov	r10d, esi
	mov	BYTE PTR [rsp+31], 0
	test	edi, edi
	jne	.L18
	mov	BYTE PTR [rsp+30], 48
	mov	r8d, 14
	jmp	.L15
.L18:
	mov	ecx, 14
	mov	esi, 10
.L14:
	mov	eax, edi
	xor	edx, edx
	mov	r8, rcx
	div	esi
	add	edx, 48
	mov	edx, edi
	dec	rcx
	mov	edi, eax
	cmp	edx, 9
	ja	.L14
.L15:
	mov	r9d, 15
	mov	edi, 1
	sub	r9d, r8d
.L16:
	jge	.L21
	mov	BYTE PTR [rsp+15], 48
	mov	eax, edi
	lea	rsi, [rsp+15]
	mov	edx, 1
	syscall
	inc	r9d
	jmp	.L16
.L21:
	movsx	r8, r8d
	lea	rdi, [rsp+16+r8]
	call	write_str
	add	rsp, 32
	ret
	.size	write_dec_padded, .-write_dec_padded
write_hex_padded:
	mov	r10d, esi
	mov	BYTE PTR [rsp+31], 0
	test	edi, edi
	jne	.L27
	mov	BYTE PTR [rsp+30], 48
	mov	r8d, 14
	jmp	.L24
.L27:
	mov	eax, 14
.L23:
	mov	edx, edi
	mov	r8, rax
	and	edx, 15
	mov	dl, BYTE PTR hex_chars.0[rdx]
	dec	rax
	shr	edi, 4
	jne	.L23
.L24:
	mov	r9d, 15
	mov	edi, 1
	sub	r9d, r8d
.L25:
	cmp	r9d, r10d
	jge	.L30
	mov	BYTE PTR [rsp+15], 48
	lea	rsi, [rsp+15]
	mov	edx, 1
	syscall
	inc	r9d
	jmp	.L25
.L30:
	movsx	r8, r8d
	lea	rdi, [rsp+16+r8]
	call	write_str
	add	rsp, 32
	ret
	.size	write_hex_padded, .-write_hex_padded
read_u32_le:
	mov	eax, DWORD PTR [rdi]
	ret
	.size	read_u32_le, .-read_u32_le
get_segment:
	add	eax, DWORD PTR mz[rip+28]
	lea	ecx, [rax+rdi*8]
	movzx	eax, BYTE PTR file_buf[rcx+1]
	mov	rdx, rcx
	sal	eax, 8
	lea	ecx, [rdx+2]
	mov	WORD PTR [rsi], ax
	movzx	eax, BYTE PTR file_buf[rcx+1]
	movzx	ecx, BYTE PTR file_buf[rcx]
	sal	eax, 8
	lea	ecx, [rdx+4]
	add	edx, 6
	movzx	eax, BYTE PTR file_buf[rcx+1]
	movzx	ecx, BYTE PTR file_buf[rcx]
	sal	eax, 8
	or	eax, ecx
	mov	WORD PTR [rsi+4], ax
	movzx	eax, BYTE PTR file_buf[rdx+1]
	movzx	edx, BYTE PTR file_buf[rdx]
	sal	eax, 8
	or	eax, edx
	mov	WORD PTR [rsi+6], ax
	ret
	.size	get_segment, .-get_segment
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC0:
	.string	"EXECUTABLE type="
.LC1:
	.string	"NE"
.LC2:
	.string	"MZ"
.LC3:
	.string	" code="
.LC4:
	.string	" data="
.LC5:
	.string	" relocs="
	.text
do_summary:
	push	rbp
	mov	edi, OFFSET FLAT:.LC0
	mov	ebp, esi
	push	rbx
	mov	ebx, edx
	sub	rsp, 16
	call	write_str
	cmp	DWORD PTR has_nehdr[rip], 0
	mov	edi, OFFSET FLAT:.LC1
	jne	.L37
.L37:
	call	write_str
	mov	edi, OFFSET FLAT:.LC3
	mov	edi, r8d
	mov	esi, 4
	call	write_dec_padded
	mov	edi, OFFSET FLAT:.LC4
	call	write_str
	mov	esi, 4
	mov	edi, ebp
	call	write_dec_padded
	mov	edi, OFFSET FLAT:.LC5
	call	write_str
	mov	edi, ebx
	mov	esi, 4
	call	write_dec_padded
	mov	eax, 1
	mov	BYTE PTR [rsp+15], 10
	lea	rsi, [rsp+15]
	mov	edi, eax
	syscall
	add	rsp, 16
	pop	rbx
	pop	rbp
	ret
	.size	do_summary, .-do_summary
	.section	.rodata.str1.1
.LC6:
	.string	"ERROR: "
	.text
error_exit:
	sub	rsp, 16
	mov	r8, rdi
	mov	edi, OFFSET FLAT:.LC6
	call	write_str
	mov	rdi, r8
	call	write_str
	mov	edi, 1
	mov	BYTE PTR [rsp+15], 10
	lea	rsi, [rsp+15]
	mov	eax, edi
	mov	edx, 1
	syscall
	mov	eax, 60
	.size	error_exit, .-error_exit
	.section	.rodata.str1.1
.LC7:
	.string	"truncated"
.LC8:
	.string	"file_not_found"
.LC9:
	.string	"bad_magic"
.LC10:
	.string	"bad_ne_offset"
.LC11:
	.string	"headers"
.LC12:
	.string	"MZ pages="
.LC13:
	.string	" entry="
.LC14:
	.string	"NE segments="
.LC15:
	.string	" modules="
.LC16:
	.string	"segments"
.LC17:
	.string	"SEG "
.LC18:
	.string	" off="
.LC19:
	.string	" len="
.LC20:
	.string	" flags="
.LC21:
	.string	" type="
.LC22:
	.string	"CODE"
.LC23:
	.string	"DATA"
.LC24:
	.string	"relocs"
.LC25:
	.string	"RELOC seg="
	.text
main_func:
	push	r14
	push	r12
	push	rbp
	cmp	edi, 2
	jg	.L41
.L46:
	jmp	.L93
.L41:
	mov	r9, QWORD PTR [rsi+16]
	mov	rdi, QWORD PTR [rsi+8]
	xor	esi, esi
	mov	eax, 2
	mov	edx, esi
	syscall
	mov	rdi, rax
	test	rax, rax
	js	.L44
	xor	esi, esi
	mov	eax, 8
	mov	edx, 2
	syscall
	mov	r8, rax
	test	rax, rax
	jns	.L43
	mov	eax, 3
	syscall
	jmp	.L44
.L43:
	mov	eax, 8
	xor	edx, edx
	syscall
	mov	rdx, r8
	mov	rax, r8
	cmp	r8, 65536
	jbe	.L45
	mov	edx, 65536
.L45:
	mov	esi, OFFSET FLAT:file_buf
	syscall
	mov	rsi, rax
	mov	eax, 3
	test	rsi, rsi
	js	.L44
	cmp	rsi, 27
	ja	.L88
	jmp	.L46
.L44:
	mov	edi, OFFSET FLAT:.LC8
.L93:
	call	error_exit
.L88:
	cmp	BYTE PTR file_buf[rip], 77
	jne	.L48
	jne	.L48
	mov	rax, QWORD PTR file_buf[rip+2]
	mov	bx, WORD PTR file_buf[rip+6]
	mov	r13w, WORD PTR file_buf[rip+22]
	mov	QWORD PTR mz[rip], rax
	mov	rax, QWORD PTR file_buf[rip+10]
	mov	r8d, DWORD PTR file_buf[rip+24]
	mov	QWORD PTR mz[rip+16], rax
	mov	ax, WORD PTR file_buf[rip+26]
	mov	WORD PTR mz[rip+24], ax
	cmp	rsi, 63
	mov	edi, OFFSET FLAT:file_buf+60
	call	read_u32_le
	mov	DWORD PTR mz[rip+28], eax
	mov	edx, eax
	test	eax, eax
	jne	.L89
	jmp	.L50
.L48:
	mov	edi, OFFSET FLAT:.LC9
	jmp	.L93
.L49:
	xor	edx, edx
	mov	DWORD PTR mz[rip+28], edx
.L50:
	xor	eax, eax
	mov	DWORD PTR has_nehdr[rip], eax
	jmp	.L52
.L89:
	lea	ecx, [rax+64]
	cmp	rsi, rcx
	jb	.L46
	mov	eax, eax
	jne	.L53
	lea	eax, [rdx+1]
	cmp	BYTE PTR file_buf[rax], 69
	jne	.L53
	mov	DWORD PTR has_nehdr[rip], 1
	lea	esi, [rdx+2]
	lea	edi, [rdx+8]
	movzx	eax, BYTE PTR file_buf[rsi+1]
	movzx	esi, BYTE PTR file_buf[rsi]
	sal	eax, 8
	or	eax, esi
	lea	esi, [rdx+4]
	mov	WORD PTR nehdr[rip], ax
	movzx	eax, BYTE PTR file_buf[rsi+1]
	movzx	esi, BYTE PTR file_buf[rsi]
	or	eax, esi
	lea	esi, [rdx+6]
	mov	WORD PTR nehdr[rip+2], ax
	movzx	eax, BYTE PTR file_buf[rsi+1]
	movzx	esi, BYTE PTR file_buf[rsi]
	sal	eax, 8
	or	eax, esi
	lea	esi, [rdx+12]
	mov	WORD PTR nehdr[rip+4], ax
	call	read_u32_le
	movzx	eax, BYTE PTR file_buf[rsi+1]
	movzx	esi, BYTE PTR file_buf[rsi]
	sal	eax, 8
	or	eax, esi
	lea	esi, [rdx+14]
	mov	WORD PTR nehdr[rip+12], ax
	movzx	eax, BYTE PTR file_buf[rsi+1]
	movzx	esi, BYTE PTR file_buf[rsi]
	sal	eax, 8
	or	eax, esi
	lea	esi, [rdx+16]
	mov	WORD PTR nehdr[rip+14], ax
	movzx	eax, BYTE PTR file_buf[rsi+1]
	movzx	esi, BYTE PTR file_buf[rsi]
	sal	eax, 8
	or	eax, esi
	lea	esi, [rdx+18]
	mov	WORD PTR nehdr[rip+16], ax
	movzx	eax, BYTE PTR file_buf[rsi+1]
	movzx	esi, BYTE PTR file_buf[rsi]
	or	eax, esi
	lea	esi, [rdx+20]
	mov	WORD PTR nehdr[rip+18], ax
	movzx	eax, BYTE PTR file_buf[rsi+1]
	movzx	esi, BYTE PTR file_buf[rsi]
	sal	eax, 8
	or	eax, esi
	lea	esi, [rdx+22]
	mov	WORD PTR nehdr[rip+20], ax
	movzx	eax, BYTE PTR file_buf[rsi+1]
	movzx	esi, BYTE PTR file_buf[rsi]
	sal	eax, 8
	or	eax, esi
	lea	esi, [rdx+24]
	mov	WORD PTR nehdr[rip+22], ax
	movzx	eax, BYTE PTR file_buf[rsi+1]
	movzx	esi, BYTE PTR file_buf[rsi]
	sal	eax, 8
	or	eax, esi
	lea	esi, [rdx+26]
	mov	WORD PTR nehdr[rip+24], ax
	movzx	eax, BYTE PTR file_buf[rsi+1]
	or	eax, esi
	lea	esi, [rdx+28]
	mov	WORD PTR nehdr[rip+26], ax
	movzx	eax, BYTE PTR file_buf[rsi+1]
	movzx	esi, BYTE PTR file_buf[rsi]
	sal	eax, 8
	or	eax, esi
	lea	esi, [rdx+30]
	mov	WORD PTR nehdr[rip+28], ax
	movzx	eax, BYTE PTR file_buf[rsi+1]
	movzx	esi, BYTE PTR file_buf[rsi]
	sal	eax, 8
	or	eax, esi
	lea	esi, [rdx+32]
	mov	WORD PTR nehdr[rip+30], ax
	movzx	eax, BYTE PTR file_buf[rsi+1]
	sal	eax, 8
	or	eax, esi
	lea	esi, [rdx+34]
	mov	WORD PTR nehdr[rip+32], ax
	movzx	eax, BYTE PTR file_buf[rsi+1]
	movzx	esi, BYTE PTR file_buf[rsi]
	or	eax, esi
	lea	esi, [rdx+36]
	mov	WORD PTR nehdr[rip+34], ax
	movzx	esi, BYTE PTR file_buf[rsi]
	sal	eax, 8
	add	rdi, OFFSET FLAT:file_buf
	or	eax, esi
	lea	esi, [rdx+38]
	mov	WORD PTR nehdr[rip+36], ax
	movzx	eax, BYTE PTR file_buf[rsi+1]
	movzx	esi, BYTE PTR file_buf[rsi]
	sal	eax, 8
	or	eax, esi
	lea	esi, [rdx+40]
	mov	WORD PTR nehdr[rip+38], ax
	movzx	eax, BYTE PTR file_buf[rsi+1]
	movzx	esi, BYTE PTR file_buf[rsi]
	or	eax, esi
	mov	WORD PTR nehdr[rip+40], ax
	movzx	eax, BYTE PTR file_buf[rsi+1]
	movzx	esi, BYTE PTR file_buf[rsi]
	sal	eax, 8
	or	eax, esi
	mov	WORD PTR nehdr[rip+42], ax
	call	read_u32_le
	lea	edi, [rdx+56]
	mov	DWORD PTR nehdr[rip+44], eax
	movzx	eax, BYTE PTR file_buf[rsi+1]
	movzx	esi, BYTE PTR file_buf[rsi]
	sal	eax, 8
	lea	esi, [rdx+50]
	mov	WORD PTR nehdr[rip+48], ax
	movzx	eax, BYTE PTR file_buf[rsi+1]
	sal	eax, 8
	or	eax, esi
	mov	WORD PTR nehdr[rip+50], ax
	movzx	eax, BYTE PTR file_buf[rsi+1]
	movzx	esi, BYTE PTR file_buf[rsi]
	sal	eax, 8
	or	eax, esi
	lea	esi, [rdx+60]
	mov	WORD PTR nehdr[rip+52], ax
	lea	eax, [rdx+54]
	mov	al, BYTE PTR file_buf[rax]
	mov	BYTE PTR nehdr[rip+54], al
	lea	eax, [rdx+55]
	add	edx, 62
	mov	BYTE PTR nehdr[rip+55], al
	call	read_u32_le
	mov	DWORD PTR nehdr[rip+56], eax
	movzx	eax, BYTE PTR file_buf[rsi+1]
	mov	WORD PTR nehdr[rip+60], ax
	movzx	eax, BYTE PTR file_buf[rdx+1]
	movzx	edx, BYTE PTR file_buf[rdx]
	sal	eax, 8
	mov	WORD PTR nehdr[rip+62], ax
	movzx	eax, BYTE PTR file_buf[rcx+1]
	movzx	edx, BYTE PTR file_buf[rcx]
	sal	eax, 8
	or	eax, edx
	mov	WORD PTR nehdr[rip+64], ax
	jmp	.L52
.L53:
	mov	edi, OFFSET FLAT:.LC10
	jmp	.L93
.L52:
	mov	esi, OFFSET FLAT:.LC11
	mov	rdi, r9
	call	my_strcmp
	mov	r12d, eax
	jne	.L54
	call	write_str
	movzx	edi, r10w
	mov	esi, 4
	call	write_dec_padded
	mov	edi, OFFSET FLAT:.LC5
	call	write_str
	movzx	edi, bx
	mov	esi, 4
	call	write_dec_padded
	call	write_str
	movzx	edi, r13w
	mov	esi, 4
	call	write_hex_padded
	mov	BYTE PTR [rsp+8], 58
	mov	eax, 1
	lea	rsi, [rsp+8]
	mov	edx, 1
	syscall
	movzx	edi, WORD PTR mz[rip+18]
	mov	esi, 4
	call	write_hex_padded
	mov	BYTE PTR [rsp+8], 10
	mov	eax, 1
	mov	edi, 1
	lea	rsi, [rsp+8]
	mov	edx, 1
	syscall
	cmp	DWORD PTR has_nehdr[rip], 0
	jne	.L55
.L58:
	xor	r9d, r9d
	xor	r8d, r8d
	jmp	.L56
.L55:
	mov	edi, OFFSET FLAT:.LC14
	call	write_str
	movzx	edi, WORD PTR nehdr[rip+28]
	call	write_dec_padded
	mov	edi, OFFSET FLAT:.LC15
	call	write_str
	movzx	edi, WORD PTR nehdr[rip+30]
	mov	esi, 2
	call	write_dec_padded
	mov	edi, OFFSET FLAT:.LC13
	call	write_str
	movzx	edi, WORD PTR nehdr[rip+22]
	mov	esi, 4
	call	write_hex_padded
	mov	BYTE PTR [rsp+8], 58
	mov	eax, 1
	mov	edi, 1
	lea	rsi, [rsp+8]
	mov	edx, 1
	syscall
	movzx	edi, WORD PTR nehdr[rip+20]
	mov	esi, 4
	mov	BYTE PTR [rsp+8], 10
	mov	eax, 1
	mov	edi, 1
	lea	rsi, [rsp+8]
	mov	edx, 1
	xor	r9d, r9d
	xor	r8d, r8d
	cmp	DWORD PTR has_nehdr[rip], 0
	je	.L58
.L57:
	movzx	eax, WORD PTR nehdr[rip+28]
	cmp	eax, r12d
	jle	.L56
	lea	rsi, [rsp+8]
	mov	edi, r12d
	call	get_segment
	movzx	eax, WORD PTR [rsp+10]
	test	BYTE PTR [rsp+12], 1
	je	.L59
	add	r8d, eax
	jmp	.L60
.L59:
	add	r9d, eax
.L60:
	inc	r12d
	jmp	.L57
.L56:
	movzx	edx, WORD PTR mz[rip+4]
	jmp	.L91
.L54:
	mov	esi, OFFSET FLAT:.LC16
	call	my_strcmp
	mov	ebp, eax
	test	eax, eax
	jne	.L63
	xor	ebx, ebx
	cmp	DWORD PTR has_nehdr[rip], 0
	je	.L64
.L65:
	cmp	ebp, eax
	jge	.L64
	lea	rsi, [rsp+8]
	movzx	r13d, WORD PTR [rsp+8]
	movzx	ecx, WORD PTR nehdr[rip+50]
	sal	r13d, cl
	call	write_str
	mov	esi, 2
	mov	edi, ebp
	call	write_dec_padded
	mov	edi, OFFSET FLAT:.LC18
	call	write_str
	mov	esi, 8
	mov	edi, r13d
	call	write_hex_padded
	mov	edi, OFFSET FLAT:.LC19
	call	write_str
	movzx	r13d, WORD PTR [rsp+10]
	mov	esi, 4
	mov	edi, r13d
	call	write_hex_padded
	mov	edi, OFFSET FLAT:.LC20
	call	write_str
	mov	esi, 4
	mov	edi, OFFSET FLAT:.LC21
	call	write_str
	and	r14d, 1
	je	.L66
	mov	edi, OFFSET FLAT:.LC22
	add	ebx, r13d
	call	write_str
	jmp	.L67
.L66:
	mov	edi, OFFSET FLAT:.LC23
	add	r12d, r13d
	call	write_str
.L67:
	mov	eax, 1
	mov	BYTE PTR [rsp+7], 10
	lea	rsi, [rsp+7]
	mov	edx, 1
	mov	edi, eax
	syscall
	inc	ebp
	jmp	.L65
.L64:
	movzx	edx, WORD PTR mz[rip+4]
	mov	esi, r12d
	mov	edi, ebx
	jmp	.L92
.L63:
	mov	esi, OFFSET FLAT:.LC24
	call	my_strcmp
	mov	r12d, eax
	test	eax, eax
	jne	.L46
	test	bx, bx
	je	.L69
	movzx	ebp, r8w
	xor	r13d, r13d
.L70:
	movzx	eax, WORD PTR mz[rip+4]
	mov	ebx, eax
	cmp	r13d, eax
	jge	.L69
	lea	eax, [rbp+4]
	cmp	QWORD PTR file_size[rip], rax
	jb	.L69
	mov	eax, ebp
	mov	edi, OFFSET FLAT:.LC25
	movzx	ebx, BYTE PTR file_buf[rax+1]
	movzx	eax, BYTE PTR file_buf[rbp]
	sal	ebx, 8
	lea	eax, [rbp+2]
	movzx	r8d, BYTE PTR file_buf[rax+1]
	movzx	eax, BYTE PTR file_buf[rax]
	sal	r8d, 8
	or	r8d, eax
	call	write_str
	movzx	edi, r8w
	mov	esi, 4
	call	write_str
	movzx	edi, bx
	mov	esi, 4
	call	write_hex_padded
	mov	BYTE PTR [rsp+8], 10
	mov	eax, 1
	lea	rsi, [rsp+8]
	mov	edx, 1
	syscall
	inc	r13d
	add	rbp, 4
	jmp	.L70
.L69:
	xor	r9d, r9d
	cmp	DWORD PTR has_nehdr[rip], 0
	je	.L73
.L72:
	movzx	eax, WORD PTR nehdr[rip+28]
	cmp	eax, r12d
	jle	.L73
	lea	rsi, [rsp+8]
	mov	edi, r12d
	call	get_segment
	movzx	eax, WORD PTR [rsp+10]
	test	BYTE PTR [rsp+12], 1
	je	.L74
	add	r8d, eax
	jmp	.L75
.L74:
	add	r9d, eax
.L75:
	inc	r12d
	jmp	.L72
.L73:
	movzx	edx, bx
.L91:
	mov	edi, r8d
.L92:
	call	do_summary
	mov	eax, 60
	syscall
	.size	main_func, .-main_func
_start:
	xor rbp, rbp
mov rdi, [rsp]
lea rsi, [rsp+8]
call main_func
mov eax, 60
xor edi, edi
syscall

	ud2
	.size	_start, .-_start
	.section	.rodata
	.align 16
	.size	hex_chars.0, 17
hex_chars.0:
	.string	"0123456789ABCDEF"
	.local	has_nehdr
	.comm	has_nehdr,4,4
	.local	nehdr
	.comm	nehdr,68,32
	.local	mz
	.comm	mz,32,32
	.local	file_size
	.comm	file_size,8,8
	.local	file_buf
	.comm	file_buf,65536,32
	.section	.note.GNU-stack,"",@progbits
